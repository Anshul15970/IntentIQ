from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from finetuning.config import MODEL_NAME, OUTPUT_DIR

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    OUTPUT_DIR
)

print("LoRA model loaded successfully!")

model.print_trainable_parameters()

messages = [
    {
        "role": "system",
        "content": "You are an intent classification assistant.\n\nReturn ONLY the intent label."
    },
    {
        "role": "user",
        "content": "I am still waiting on my card."
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
    max_new_tokens=20
)

prediction = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\nPrediction:")
print(prediction)