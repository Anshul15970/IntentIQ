
import os
import requests

import gradio as gr

from utils.load_benchmark import load_benchmark
from utils.load_error_summary import load_error_summary
from utils.error_summary import summarize_errors
from utils.project_summary import generate_project_summary
from prompts.zero_shot_prompt import build_zero_shot_prompt
from prompts.few_shot_prompt import build_few_shot_prompt

import time
import os


port = int(os.environ.get("PORT", 7860))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def compare_models(query):
    
    if not query or not query.strip():
        return [["Error", "❌ Please enter a banking query.", ""]]

    results = []

    for model_name in [
        "Zero-shot",
        "Few-shot"
    ]:

        prediction, inference_time = predict_intent(
            query,
            model_name
        )

        results.append([
            model_name,
            prediction,
            inference_time
        ])

    return results

def predict_intent(query, selected_model):

    if not query or not query.strip():
        return "❌ Please enter a banking query.", ""

    start = time.time()

    if selected_model == "Zero-shot":

        system_prompt = build_zero_shot_prompt()

    elif selected_model == "Few-shot":

        system_prompt = build_few_shot_prompt(
            num_examples=5
        )

    else:

        return "Unknown model", ""

    prediction = call_hf_api(
        system_prompt,
        query
    )

    elapsed = time.time() - start

    return prediction, f"{elapsed:.3f} seconds"

def load_confusion_matrix(model):

    mapping = {
        "Zero-shot": "results/zero_shot_confusion_matrix.png",
        "Few-shot": "results/few_shot_confusion_matrix.png",
        "Dynamic Few-shot": "results/dynamic_few_shot_confusion_matrix.png",
        "LoRA": "results/lora_confusion_matrix.png"
    }

    return mapping[model]

def call_hf_api(system_prompt, user_prompt):

    payload = {
        "model": "qwen/qwen3-32b",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "max_tokens": 20,
        "temperature": 0,
        "reasoning_effort": "none"
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    if not response.ok:
        print(response.status_code)
        print(response.text)
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"].strip()

with gr.Blocks(title="IntentIQ") as demo:

    with gr.Tab("Prediction"):

        # Move your existing UI here

        gr.Markdown("# 🏦 IntentIQ")
        gr.Markdown("Banking77 Intent Classification Demo")

        query = gr.Textbox(
            label="Enter your banking query"
        )

        dropdown = gr.Dropdown(
            choices=[
                "Zero-shot",
                "Few-shot"
            ],
            value="Zero-shot",
            label="Model"
        )

        output = gr.Textbox(label="Prediction")

        time_output = gr.Textbox(
            label="Inference Time",
            interactive=False
        )

        comparison_table = gr.Dataframe(
        headers=["Model", "Prediction", "Inference Time"],
        datatype=["str", "str", "str"],
        label="Model Comparison"
    )
        
        button = gr.Button("Predict")
        compare_button = gr.Button("Compare All Models")

        button.click(
        fn=predict_intent,
        inputs=[query, dropdown],
        outputs=[output, time_output]
    )
        
        compare_button.click(
        fn=compare_models,
        inputs=query,
        outputs=comparison_table
    )

    with gr.Tab("Benchmark"):

        gr.Markdown("## Model Performance")

        benchmark_table = gr.Dataframe(
            value=load_benchmark(),
            label="Benchmark Results"
        )
    
    with gr.Tab("Confusion Matrix"):

        cm_dropdown = gr.Dropdown(
            choices=[
                "Zero-shot",
                "Few-shot",
                "Dynamic Few-shot",
                "LoRA"
            ],
            value="LoRA",
            label="Model"
        )

        cm_image = gr.Image(
            value="results/lora_confusion_matrix.png",
            label="Confusion Matrix"
        )
    
    with gr.Tab("Error Analysis"):
        error_dropdown = gr.Dropdown(
            choices=[
                "Zero-shot",
                "Few-shot",
                "Dynamic Few-shot",
                "LoRA"
            ],
            value="LoRA",
            label="Model"
        )

        error_table = gr.Dataframe(
            value=load_error_summary("LoRA"),
            label="Top Error Summary"
        )
        
        summary_box = gr.Textbox(
            label="Automatic Error Summary",
            lines=12
        )
        
        summary_box.value = summarize_errors("LoRA")
        
        gr.Markdown("## Overall Project Analysis")

        project_summary = gr.Textbox(
            value=generate_project_summary(),
            lines=18,
            interactive=False,
            label="Comparative Analysis"
        )
            
    cm_dropdown.change(
        fn=load_confusion_matrix,
        inputs=cm_dropdown,
        outputs=cm_image
    )

    error_dropdown.change(
        fn=lambda x: (
            load_error_summary(x),
            summarize_errors(x)
        ),
        inputs=error_dropdown,
        outputs=[
            error_table,
            summary_box
        ]
    )




demo.launch(
    server_name="0.0.0.0",
    server_port=port
)