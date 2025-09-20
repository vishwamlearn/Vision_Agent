import gradio as gr
import cv2
import supervision as sv
import base64
import operator
from typing import TypedDict, Annotated
import numpy as np

# LangGraph and LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Autodistill and GroundingDINO imports
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from dotenv import load_dotenv
load_dotenv()
import os

# Hugging Face imports for Gemma
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
from io import BytesIO

# --- Define the State of the Graph ---
class AgentState(TypedDict):
    """Represents the state of the agent's workflow."""
    messages: Annotated[list[BaseMessage], operator.add]
    image_data: bytes
    user_request: str
    current_prompt: str
    final_image: np.ndarray

# --- Load the Gemma 3 4B-IT model globally ---
# This ensures the model is loaded once at the start, not on every call
try:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    model_id = "google/gemma-3-4b-it"
    gemma_processor = AutoProcessor.from_pretrained(model_id)
    gemma_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
except Exception as e:
    print(f"Failed to load Gemma model: {e}")
    gemma_processor = None
    gemma_model = None

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
    text_scale=3.5,  # Increase text size (default is 0.5)
    text_thickness=2, # Increase text thickness (default is 1)
    # You might also adjust text_color or text_padding if needed
).annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )
    
    _, buffer = cv2.imencode('.jpg', annotated_image_with_labels)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    print(f"GroundingDINO ran with prompt: '{current_prompt}'")

    return {
        "messages": [HumanMessage(content=[
            {"type": "text", "text": "Here is the annotated image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])],
        "final_image": annotated_image_with_labels
    }

def gemma_analysis_node(state: AgentState):
    """Node to have Gemma analyze the annotated image and provide feedback."""
    if gemma_model is None or gemma_processor is None:
        raise RuntimeError("Gemma model not loaded. Please ensure you have the correct setup.")

    image_data = state['image_data']
    user_request = state['user_request']
    current_prompt = state['current_prompt']
    
    # Convert image bytes to a PIL Image object
    image_pil = Image.open(BytesIO(image_data))

    vision_prompt = (
        f"The user wants to find: '{user_request}'. I used the detection prompt "
        f"'{current_prompt}' and the result is the annotated image provided. "
        "Critique the results. Are all relevant objects found? Are there false positives? "
        "If the detection is satisfactory, respond with only the word 'SATISFIED'. "
        "If the detection needs improvement, provide a more refined, comma-separated detection prompt to improve the results."
    )
    
    # The Gemma model requires a specific chat format with the image
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": vision_prompt},
            {"type": "image", "image": image_pil}
        ]}
    ]

    input_tensors = gemma_processor(messages, return_tensors="pt").to(device)
    response_tokens = gemma_model.generate(**input_tensors, max_new_tokens=100)
    response_text = gemma_processor.decode(response_tokens[0], skip_special_tokens=True)
    
    print(f"Gemma response: {response_text}")

    if "SATISFIED" in response_text.upper():
        return {"current_prompt": None, "messages": [HumanMessage(content="SATISFIED")]}
    else:
        return {"current_prompt": response_text, "messages": [HumanMessage(content=response_text)]}

# --- Define the Router ---
def decide_to_continue(state: AgentState):
    """Router node to decide if the workflow should continue or end."""
    if state['current_prompt'] is None:
        return "end"
    else:
        return "run_grounding_dino"

# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("run_grounding_dino", run_grounding_dino_node)
workflow.add_node("gemma_analysis", gemma_analysis_node)

workflow.add_edge(START, "run_grounding_dino")
workflow.add_edge("run_grounding_dino", "gemma_analysis")
workflow.add_conditional_edges("gemma_analysis", decide_to_continue, {"run_grounding_dino": "run_grounding_dino", "end": END})

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
        "current_prompt": request
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
    title="LangGraph-powered Image Detection Agent with Gemma 3",
    description="Upload an image and specify what to detect. The agent will use Gemma 3 to refine its search until the detection is satisfactory."
)

if __name__ == "__main__":
    iface.launch()