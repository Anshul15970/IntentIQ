import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model
)

from trl import SFTTrainer

from finetuning.config import *

from finetuning.data import prepare_dataset

print("Loading dataset...")

train_dataset, test_dataset, label_names = prepare_dataset()

print(f"Training examples : {len(train_dataset)}")
print(f"Testing examples  : {len(test_dataset)}")

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)

print("Model loaded.")

print("Attaching LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters() 

print("Creating training arguments...")

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    logging_steps=LOGGING_STEPS,

    save_steps=SAVE_STEPS,

    report_to="none",

    remove_unused_columns=False,

    fp16=False,

    bf16=False
)
 
print("Creating trainer...")

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    processing_class=tokenizer,

    formatting_func=lambda example: example["text"]
)

print("Starting training...")

trainer.train()

print("Saving adapter...")

trainer.save_model(OUTPUT_DIR)

print("Training complete!")