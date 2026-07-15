import gradio as gr

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from utils.load_benchmark import load_benchmark
from peft import PeftModel
from utils.load_error_summary import load_error_summary
from finetuning.config import MODEL_NAME
from finetuning.config import OUTPUT_DIR
from models.prompt_models.qwen_model import QwenModel
from utils.error_summary import summarize_errors
from utils.project_summary import generate_project_summary

print("Loading LoRA model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    OUTPUT_DIR
)

print("LoRA loaded successfully!")

# Prompt models (lazy loaded)

zero_model = None
few_model = None
dynamic_model = None

def get_prompt_model(model_type):

    global zero_model
    global few_model
    global dynamic_model

    if model_type == "Zero-shot":

        if zero_model is None:
            print("Loading Zero-shot model...")
            zero_model = QwenModel(prompt_type="zero_shot")
            zero_model.load_model()

        return zero_model

    elif model_type == "Few-shot":

        if few_model is None:
            print("Loading Few-shot model...")
            few_model = QwenModel(prompt_type="few_shot")
            few_model.load_model()

        return few_model

    elif model_type == "Dynamic Few-shot":

        if dynamic_model is None:
            print("Loading Dynamic Few-shot model...")
            dynamic_model = QwenModel(
                prompt_type="dynamic_few_shot"
            )
            dynamic_model.load_model()

        return dynamic_model

    return None

import time

def compare_models(query):
    
    if not query or not query.strip():
        return [["Error", "❌ Please enter a banking query.", ""]]

    results = []

    for model_name in [
        "Zero-shot",
        "Few-shot",
        "Dynamic Few-shot",
        "LoRA"
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

    if selected_model == "LoRA":

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intent classification assistant.\n\n"
                    "Return ONLY the intent label."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        temperature=None,
        top_p=None
        )

        prediction = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

    else:

        prompt_model = get_prompt_model(selected_model)

        result = prompt_model.predict(query)

        prediction = result["prediction"]

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
                "LoRA",
                "Zero-shot",
                "Few-shot",
                "Dynamic Few-shot"
            ],
            value="LoRA",
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
    server_port=7860
)