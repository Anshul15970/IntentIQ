import time

import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from peft import PeftModel

from finetuning.config import *
from finetuning.data import prepare_dataset

def predict_intent(model, tokenizer, text):

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
            "content": text
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

    start = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=20
    )

    elapsed = time.time() - start

    prediction = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    return prediction, elapsed

print("Loading dataset...")

train_dataset, test_dataset, label_names = prepare_dataset(
    format_for_training=False
)
print(f"Test examples: {len(test_dataset)}")

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)

print("Loading second base model for LoRA...")

lora_base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    lora_base_model,
    OUTPUT_DIR
)

print("Models ready!")

predictions = []

true_labels = []

texts = []

times = []

correct = []

for i, sample in enumerate(test_dataset):

    prediction, inference_time = predict_intent(
        model,
        tokenizer,
        sample["text"]
    )

    predictions.append(prediction)

    true_labels.append(
        label_names[sample["label"]]
    )

    texts.append(sample["text"])

    times.append(inference_time)

    correct.append(
        prediction == label_names[sample["label"]]
    )
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(test_dataset)} samples...")

accuracy = accuracy_score(
    true_labels,
    predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)

average_time = sum(times) / len(times)

print("\n" + "=" * 50)
print("Evaluation Results")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Avg Time : {average_time:.4f} sec")

import os

os.makedirs("results", exist_ok=True)

# -----------------------------
# Save all predictions
# -----------------------------

results_df = pd.DataFrame({
    "text": texts,
    "true_label": true_labels,
    "prediction": predictions,
    "correct": correct,
    "inference_time": times
})

results_df.to_csv(
    "results/lora_predictions.csv",
    index=False
)

print("\nSaved predictions to results/lora_predictions.csv")


# -----------------------------
# Save only errors
# -----------------------------

errors_df = results_df[
    results_df["correct"] == False
]

errors_df.to_csv(
    "results/lora_errors.csv",
    index=False
)

print(f"Saved {len(errors_df)} errors to results/lora_errors.csv")


# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=label_names
)

plt.figure(figsize=(22, 22))

plt.imshow(cm, interpolation="nearest")

plt.title("LoRA Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(len(label_names)),
    label_names,
    rotation=90,
    fontsize=6
)

plt.yticks(
    range(len(label_names)),
    label_names,
    fontsize=6
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "results/lora_confusion_matrix.png",
    dpi=300
)

plt.close()

print("Saved confusion matrix to results/lora_confusion_matrix.png")