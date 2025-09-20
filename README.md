# 🤖 LangGraph-powered Image Detection Agent

This project demonstrates a powerful, autonomous agent for image detection using a combination of **LangGraph**, **GroundingDINO**, and a **multimodal Large Language Model (LLM)**. The agent intelligently refines its object detection prompts based on feedback from the LLM, ensuring accurate and reliable results.

A user-friendly web interface is provided using **Gradio**, allowing you to upload an image and specify what you want to detect.

<br>

## ✨ Features

- **Autonomous Agent**: An intelligent agent that uses a feedback loop to improve detection.
- **Dynamic Prompting**: The agent can refine its detection prompts (e.g., from "people" to "men, lady") based on visual analysis from the LLM.
- **Multimodal Integration**: Leverages a vision-capable LLM (GPT-4o or Gemma 3) to analyze annotated images and provide actionable feedback.
- **Gradio UI**: A simple, intuitive web interface for uploading images and specifying detection requests.
- **Open-Source Stack**: Built on top of popular open-source libraries like LangGraph, Autodistill, and Supervision.

<br>

## 🚀 Quick Start

Follow these steps to get the project up and running.

### Prerequisites

1.  **Python 3.9+**
2.  **A GPU with at least 8GB of VRAM** is recommended for running the `GroundingDINO` model.
3.  **An OpenAI API Key** for GPT-4o, or a powerful GPU for running Gemma 3 locally.

DEMO:-
<img width="1330" height="684" alt="image" src="https://github.com/user-attachments/assets/c3e1b938-c27d-4cf2-83da-0f39c1354e44" />

