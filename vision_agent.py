import gradio as gr
import cv2
import supervision as sv
import base64
import operator
from typing import TypedDict, Annotated
import numpy as np

# LangGraph and LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Autodistill and GroundingDINO imports
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Define the State of the Graph ---
class AgentState(TypedDict):
    """Represents the state of the agent's workflow."""
    messages: Annotated[list[BaseMessage], operator.add]
    image_data: bytes
    user_request: str
    current_prompt: str
    final_image: np.ndarray
    detected_labels: list[str]  # New key to store the labels

# --- Define the Nodes ---
def run_grounding_dino_node(state: AgentState):
    """Node to run GroundingDINO and annotate the image."""
    image_data = state['image_data']
    current_prompt = state['current_prompt']

    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    prompts = [p.strip() for p in current_prompt.split(',')]
    ontology_dict = {prompt: prompt for prompt in prompts}
    ontology = CaptionOntology(ontology_dict)

    base_model = GroundingDINO(ontology=ontology)
    predictions = base_model.predict(image)

    detections = sv.Detections(
        xyxy=predictions.xyxy,
        class_id=predictions.class_id,
        confidence=predictions.confidence,
    )

    class_names = ontology.prompts()
    labels = [f"{class_names[class_id]}: {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]

    annotated_image = sv.BoxAnnotator().annotate(scene=image.copy(), detections=detections)
    annotated_image_with_labels = sv.LabelAnnotator(
        text_scale=3.5, 
        text_thickness=2
    ).annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )
    
    _, buffer = cv2.imencode('.jpg', annotated_image_with_labels)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    print(f"GroundingDINO ran with prompts: '{current_prompt}'")

    # Update state with the final image, base64 data, and the new labels list
    return {
        "messages": [HumanMessage(content=[
            {"type": "text", "text": "Here is the annotated image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])],
        "final_image": annotated_image_with_labels,
        "detected_labels": labels # Pass the detected labels to the state
    }

def gpt4o_analysis_node(state: AgentState):
    """Node to have GPT-4o analyze the annotated image and provide feedback."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=100)
    
    messages = state['messages']
    user_request = state['user_request']
    current_prompt = state['current_prompt']
    detected_labels = state['detected_labels']

    vision_prompt = (
        "The image contains bounding boxes labeled with a detected class and confidence to specific objects.\n"
        "Your task is to identify the objects indicated by these bounding boxes and determine whether each detected object is relevant to the user's query.\n"
        f"The user's initial request was: '{user_request}'. "
        f"The current detection prompt is: '{current_prompt}'. "
        f"The annotated image shows that the model detected the following objects: {', '.join(detected_labels)}. "
        f"Critique the detection results. Does the list of detected objects completely and accurately fulfill the user's request? "
        "Consider if there are any objects that were missed or if there are irrelevant or inaccurate detections. "
        "If the detection is satisfactory, respond with only the word 'SATISFIED'. "
        "If the detection needs improvement, provide a more refined, comma-separated detection prompt to improve the results."
    )
    
    messages.append(SystemMessage(content=vision_prompt))
    
    response = llm.invoke(messages)
    response_text = response.content
    
    print(f"GPT-4o response: {response_text}")
    
    if "SATISFIED" in response_text.upper():
        return {"current_prompt": None, "messages": [HumanMessage(content="SATISFIED")]}
    else:
        return {"current_prompt": response_text, "messages": [HumanMessage(content=response_text)]}

def decide_to_continue(state: AgentState):
    """Router node to decide if the workflow should continue or end."""
    if state['current_prompt'] is None:
        return "end"
    else:
        return "run_grounding_dino"
# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("run_grounding_dino", run_grounding_dino_node)
workflow.add_node("gpt4o_analysis", gpt4o_analysis_node)

workflow.add_edge(START, "run_grounding_dino")
workflow.add_edge("run_grounding_dino", "gpt4o_analysis")
workflow.add_conditional_edges("gpt4o_analysis", decide_to_continue, {"run_grounding_dino": "run_grounding_dino", "end": END})

app = workflow.compile()

# --- Gradio Integration ---
def process_image_and_request(image: np.ndarray, request: str):
    """Gradio-facing function to run the LangGraph agent."""
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    initial_state = {
        "messages": [],
        "image_data": image_bytes,
        "user_request": request,
        "current_prompt": request,
        "final_image": None, # Will be set in the first node
        "detected_labels": [] # Will be set in the first node
    }

    final_state = app.invoke(initial_state)

    return final_state['final_image']

# Set up the Gradio interface
iface = gr.Interface(
    fn=process_image_and_request,
    inputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Textbox(label="Detection Request")
    ],
    outputs=gr.Image(label="Annotated Image"),
    title="Let's see Image Detection Agent",
    description="Upload an image and specify what to detect. The agent will use GPT-4o to refine its search until the detection is satisfactory."
)

if __name__ == "__main__":
    iface.launch()
