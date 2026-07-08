from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from prompts.few_shot_prompt import build_few_shot_prompt

from models.base_model import BaseModel


class QwenModel(BaseModel):

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def predict(self, text: str):

        messages = [
            {
                "role": "system",
                "content": build_few_shot_prompt(num_examples=5)
            },
            {
                "role": "user",
                "content": text
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=20
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        prediction = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return {
            "model": self.model_name,
            "prediction": prediction
        }

    def get_model_name(self):
        return self.model_name