import cv2
import mediapipe as mp
import numpy as np
import base64
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
import gradio as gr

# --- Define the Driver Analysis Tool and its dependencies ---

# This is the original tool provided by the user, wrapped with the @tool decorator.
# It is a self-contained function that performs both Mediapipe and PaliGemma analysis.

# Initialize models and processors once
model_id = "google/paligemma-3b-pt-224"
paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
).eval()
paligemma_processor = PaliGemmaProcessor.from_pretrained(model_id)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)


@tool
def analyze_driver_behavior(image_data: bytes, request: str) -> dict:
    """
    Analyzes a driver's behavior by combining head pose analysis with PaliGemma's scene understanding,
    returning a structured JSON output.
    """
    try:
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from base64 data.")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

    # 1. Run Mediapipe Head Pose Analysis
    head_pose_results = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    drowsiness_detected = "no"
    driver_looking_aside = "no"

    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                h, w, _ = image.shape
                landmarks.append((int(landmark.x * w), int(landmark.y * h), landmark.z))

            if len(landmarks) >= 264:
                nose_tip = landmarks[1]
                chin = landmarks[152]
                left_eye = landmarks[33]
                right_eye = landmarks[263]
                
                roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                pitch = np.arctan2(chin[1] - nose_tip[1], chin[2] - nose_tip[2])
                yaw = np.arctan2(nose_tip[0] - chin[0], nose_tip[2] - chin[2])
                
                head_pose_results = {
                    "roll": np.degrees(roll),
                    "pitch": np.degrees(pitch),
                    "yaw": np.degrees(yaw)
                }

                if abs(landmarks[386][1] - landmarks[374][1]) < 5:
                    drowsiness_detected = "yes"
                
                if abs(head_pose_results['yaw']) > 15:
                    driver_looking_aside = "yes"

    # 2. Formulate a structured prompt for PaliGemma
    paligemma_prompt = f"""
    The driver's head pose is: yaw={head_pose_results['yaw']:.2f}, pitch={head_pose_results['pitch']:.2f}, roll={head_pose_results['roll']:.2f}.
    Analyze the image and respond with a JSON object.
    The JSON should contain:
    - a 'drowsiness_detected' key with a value of 'yes' or 'no'.
    - a 'driver_looking_aside' key with a value of 'yes' or 'no'.
    - a 'distraction_detected' key, listing any distracting objects detected (e.g., 'phone', 'food', 'drink'). If none, use an empty list.
    - a 'summary' key providing a final conclusion about the driver's behavior.
    """

    # 3. Run PaliGemma analysis
    model_inputs = paligemma_processor(
        text=paligemma_prompt,
        images=image_rgb,
        return_tensors="pt"
    ).to(torch.bfloat16).to(paligemma_model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = paligemma_model.generate(**model_inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]
        decoded_text = paligemma_processor.decode(generation, skip_special_tokens=True)
    
    # 4. Parse the output into JSON
    try:
        json_output = json.loads(decoded_text.strip())
        json_output["head_pose"] = head_pose_results
        return json_output
    except json.JSONDecodeError:
        print("Warning: PaliGemma output was not valid JSON. Attempting text-based parsing.")
        return {
            "drowsiness_detected": drowsiness_detected,
            "driver_looking_aside": driver_looking_aside,
            "distraction_detected": ["phone"] if "phone" in decoded_text.lower() else [],
            "summary": decoded_text,
            "head_pose": head_pose_results
        }

# --- LangGraph Setup ---

# Define the state for our graph as a Pydantic BaseModel.
class DriverState(BaseModel):
    image_data: bytes
    request: str
    result: dict | None = None

# Define the node that will execute the tool.
def analysis_node(state: DriverState):
    print("🚦 Starting driver behavior analysis...")
    
    # Call the tool with the data from the current state.
    analysis_result = analyze_driver_behavior.invoke({
        "image_data": state.image_data,
        "request": state.request
    })
    
    # Update the state with the result.
    state.result = analysis_result
    print("✅ Analysis completed. Updating state.")
    return state

# --- Gradio and LangGraph Integration ---

# Build and compile the LangGraph app once
graph_builder = StateGraph(DriverState)
graph_builder.add_node("analyze_behavior", analysis_node)
graph_builder.add_edge(START, "analyze_behavior")
graph_builder.add_edge("analyze_behavior", END)
app = graph_builder.compile()

# Function to be used by Gradio
def analyze_with_gradio(image_np, request):
    """
    Processes an image and analysis request through the LangGraph workflow.
    `image_np` is a numpy array from the Gradio Image component.
    """
    if image_np is None:
        return {"error": "No image provided."}
    
    # Convert the numpy array to a base64 string
    _, buffer = cv2.imencode('.jpg', image_np)
    base64_image_data = base64.b64encode(buffer).decode('utf-8')
    
    # Define the initial state for the LangGraph invocation
    initial_state = DriverState(
        image_data=base64_image_data,
        request=request
    )
    
    # Invoke the workflow and get the final state
    final_state = app.invoke(initial_state)
    
    # Return the analysis result
    return final_state.result

# --- Main Gradio Interface ---
if __name__ == "__main__":
    # Define the Gradio interface
    interface = gr.Interface(
        fn=analyze_with_gradio,
        inputs=[
            gr.Image(type="numpy", label="Upload Driver Image"),
            gr.Textbox(label="Analysis Request", value="Analyze the driver for signs of drowsiness and distractions.")
        ],
        outputs=gr.Json(label="Analysis Result"),
        title="Driver Behavior Analysis",
        description="Upload an image to analyze a driver's head pose, drowsiness, and distractions using Mediapipe and PaliGemma.",
        allow_flagging="never"
    )

    # Launch the interface
    interface.launch()

