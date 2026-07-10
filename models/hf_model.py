from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from models.base_model import BaseModel
from prompts.few_shot_prompt import build_few_shot_prompt
from prompts.zero_shot_prompt import build_zero_shot_prompt
from prompts.dynamic_few_shot import DynamicFewShot

DYNAMIC_RETRIEVER = DynamicFewShot()
class HFModel(BaseModel):

    def __init__(self, model_name, prompt_type="few_shot"):
        self.model_name = model_name
        self.prompt_type = prompt_type

        self.tokenizer = None

        self.model = None

    def load_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def predict(self, text: str):

        if self.prompt_type == "zero_shot":

            system_prompt = build_zero_shot_prompt()

        elif self.prompt_type == "few_shot":

            system_prompt = build_few_shot_prompt(
                num_examples=5
            )

        elif self.prompt_type == "dynamic_few_shot":

            system_prompt = DYNAMIC_RETRIEVER.build_prompt(
                text
            )

        else:

            raise ValueError(
                f"Unknown prompt type: {self.prompt_type}"
            )

        messages = [
            {
                "role": "system",
                "content": system_prompt
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