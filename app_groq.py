import time
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from finetuning.config import MODEL_NAME
from finetuning.config import OUTPUT_DIR

from models.groq_model import GroqModel

from utils.load_benchmark import load_benchmark
from utils.load_error_summary import load_error_summary
from utils.error_summary import summarize_errors
from utils.project_summary import generate_project_summary


# ============================================================
# LOAD LoRA MODEL
# ============================================================

print("Loading LoRA model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)

lora_model = PeftModel.from_pretrained(
    base_model,
    OUTPUT_DIR
)

print("LoRA loaded successfully!")


# ============================================================
# GROQ MODELS
# ============================================================

GROQ_MODELS = {
    "GPT-OSS 20B": "openai/gpt-oss-20b",
    "Qwen 3.6 27B": "qwen/qwen3.6-27b",
    "Qwen 3.8 27B": "qwen/qwen3.8-27b",
    "GPT-OSS 120B": "openai/gpt-oss-120b"
}


# ============================================================
# ALL MODELS
# ============================================================

MODELS = [
    "LoRA",
    "GPT-OSS 20B",
    "Qwen 3.6 27B",
    "Qwen 3.8 27B",
    "GPT-OSS 120B"
]


# ============================================================
# PROMPTING METHODS
# ============================================================

PROMPTING_METHODS = [
    "Zero-shot",
    "Few-shot",
    "Dynamic Few-shot"
]

PROMPT_TYPE_MAP = {
    "Zero-shot": "zero_shot",
    "Few-shot": "few_shot",
    "Dynamic Few-shot": "dynamic_few_shot"
}


# ============================================================
# GROQ MODEL CACHE
# ============================================================

groq_model_cache = {}


def get_groq_model(model_name, prompting_method):

    model_id = GROQ_MODELS[model_name]

    prompt_type = PROMPT_TYPE_MAP[
        prompting_method
    ]

    cache_key = (
        model_id,
        prompt_type
    )

    if cache_key not in groq_model_cache:

        print(
            f"Creating Groq model: "
            f"{model_name} | {prompting_method}"
        )

        groq_model_cache[cache_key] = GroqModel(
            model_name=model_id,
            prompt_type=prompt_type
        )

        groq_model_cache[cache_key].load_model()

    return groq_model_cache[cache_key]


# ============================================================
# LoRA PREDICTION
# ============================================================

def predict_lora(query):

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
    ).to(lora_model.device)

    outputs = lora_model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        temperature=None,
        top_p=None
    )

    prediction = tokenizer.decode(
        outputs[0][
            inputs["input_ids"].shape[1]:
        ],
        skip_special_tokens=True
    ).strip()

    return prediction


# ============================================================
# SINGLE PREDICTION
# ============================================================

def predict_intent(
    query,
    selected_model,
    prompting_method
):

    if not query or not query.strip():

        return (
            "❌ Please enter a banking query.",
            ""
        )

    start = time.time()

    try:

        # ====================================================
        # LoRA
        # ====================================================

        if selected_model == "LoRA":

            prediction = predict_lora(query)


        # ====================================================
        # GROQ
        # ====================================================

        else:

            groq_model = get_groq_model(
                selected_model,
                prompting_method
            )

            result = groq_model.predict(query)

            prediction = result["prediction"]


    except Exception as e:

        prediction = f"❌ Error: {str(e)}"


    elapsed = time.time() - start

    return (
        prediction,
        f"{elapsed:.3f} seconds"
    )


# ============================================================
# COMPARE ALL MODELS
# ============================================================

def compare_models(
    query,
    prompting_method
):

    if not query or not query.strip():

        return [[
            "Error",
            "❌ Please enter a banking query.",
            ""
        ]]

    results = []

    # ========================================================
    # LoRA
    # ========================================================

    prediction, inference_time = predict_intent(
        query,
        "LoRA",
        prompting_method
    )

    results.append([
        "LoRA",
        prediction,
        inference_time
    ])


    # ========================================================
    # GROQ MODELS
    # ========================================================

    for model_name in GROQ_MODELS.keys():

        prediction, inference_time = predict_intent(
            query,
            model_name,
            prompting_method
        )

        results.append([
            model_name,
            prediction,
            inference_time
        ])

    return results


# ============================================================
# CONFUSION MATRIX
# ============================================================

def load_confusion_matrix(model):

    mapping = {

        "Zero-shot":
            "results/zero_shot_confusion_matrix.png",

        "Few-shot":
            "results/few_shot_confusion_matrix.png",

        "Dynamic Few-shot":
            "results/dynamic_few_shot_confusion_matrix.png",

        "LoRA":
            "results/lora_confusion_matrix.png"
    }

    return mapping[model]


# ============================================================
# ERROR ANALYSIS
# ============================================================

def load_error_analysis(model):

    return (
        load_error_summary(model),
        summarize_errors(model)
    )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="IntentIQ - Groq"
) as demo:


    # ========================================================
    # PREDICTION
    # ========================================================

    with gr.Tab("Prediction"):

        gr.Markdown("# 🏦 IntentIQ")

        gr.Markdown(
            "### Comparative Banking Intent Classification"
        )

        # ----------------------------------------------------
        # Query
        # ----------------------------------------------------

        query = gr.Textbox(
            label="Enter your banking query",
            placeholder=(
                "e.g. I don't recognize a transaction "
                "on my card"
            ),
            lines=3
        )


        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model_dropdown = gr.Dropdown(
            choices=MODELS,
            value="LoRA",
            label="Model"
        )


        # ----------------------------------------------------
        # Prompting Strategy
        # ----------------------------------------------------

        prompting_dropdown = gr.Dropdown(
            choices=PROMPTING_METHODS,
            value="Zero-shot",
            label="Prompting Strategy"
        )


        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        with gr.Row():

            predict_button = gr.Button(
                "🚀 Predict",
                variant="primary"
            )

            compare_button = gr.Button(
                "🔍 Compare All Models"
            )


        # ----------------------------------------------------
        # Prediction Output
        # ----------------------------------------------------

        output = gr.Textbox(
            label="Predicted Intent"
        )

        time_output = gr.Textbox(
            label="Inference Time",
            interactive=False
        )


        # ----------------------------------------------------
        # Comparison
        # ----------------------------------------------------

        comparison_table = gr.Dataframe(
            headers=[
                "Model",
                "Prediction",
                "Inference Time"
            ],
            datatype=[
                "str",
                "str",
                "str"
            ],
            label="Model Comparison"
        )


        # ----------------------------------------------------
        # Predict Button
        # ----------------------------------------------------

        predict_button.click(
            fn=predict_intent,
            inputs=[
                query,
                model_dropdown,
                prompting_dropdown
            ],
            outputs=[
                output,
                time_output
            ]
        )


        # ----------------------------------------------------
        # Compare Button
        # ----------------------------------------------------

        compare_button.click(
            fn=compare_models,
            inputs=[
                query,
                prompting_dropdown
            ],
            outputs=comparison_table
        )


    # ========================================================
    # BENCHMARK
    # ========================================================

    with gr.Tab("Benchmark"):

        gr.Markdown(
            "## 📊 Model Performance"
        )

        benchmark_table = gr.Dataframe(
            value=load_benchmark(),
            label="Benchmark Results"
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    with gr.Tab("Confusion Matrix"):

        gr.Markdown(
            "## 🔥 Confusion Matrix Analysis"
        )

        cm_dropdown = gr.Dropdown(
            choices=[
                "Zero-shot",
                "Few-shot",
                "Dynamic Few-shot",
                "LoRA"
            ],
            value="LoRA",
            label="Model / Approach"
        )

        cm_image = gr.Image(
            value="results/lora_confusion_matrix.png",
            label="Confusion Matrix"
        )

        cm_dropdown.change(
            fn=load_confusion_matrix,
            inputs=cm_dropdown,
            outputs=cm_image
        )


    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    with gr.Tab("Error Analysis"):

        gr.Markdown(
            "## ⚠️ Error Analysis"
        )

        error_dropdown = gr.Dropdown(
            choices=[
                "Zero-shot",
                "Few-shot",
                "Dynamic Few-shot",
                "LoRA"
            ],
            value="LoRA",
            label="Model / Approach"
        )

        error_table = gr.Dataframe(
            value=load_error_summary("LoRA"),
            label="Top Error Summary"
        )

        summary_box = gr.Textbox(
            value=summarize_errors("LoRA"),
            label="Automatic Error Summary",
            lines=12
        )

        gr.Markdown(
            "## 📈 Overall Project Analysis"
        )

        project_summary = gr.Textbox(
            value=generate_project_summary(),
            lines=18,
            interactive=False,
            label="Comparative Analysis"
        )

        error_dropdown.change(
            fn=load_error_analysis,
            inputs=error_dropdown,
            outputs=[
                error_table,
                summary_box
            ]
        )


# ============================================================
# LAUNCH
# ============================================================

demo.launch()