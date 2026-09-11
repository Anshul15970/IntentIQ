import os
import time

from groq import Groq
from dotenv import load_dotenv

from models.base_model import BaseModel
from prompts.few_shot_prompt import build_few_shot_prompt
from prompts.zero_shot_prompt import build_zero_shot_prompt
from prompts.dynamic_few_shot import DynamicFewShot


load_dotenv()

DYNAMIC_RETRIEVER = None


class GroqModel(BaseModel):

    def __init__(self, model_name, prompt_type="few_shot"):

        self.model_name = model_name
        self.prompt_type = prompt_type

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def _build_prompt(self, text: str):

        global DYNAMIC_RETRIEVER

        if self.prompt_type == "zero_shot":

            system_prompt = build_zero_shot_prompt()

        elif self.prompt_type == "few_shot":

            system_prompt = build_few_shot_prompt(
                num_examples=5
            )

        elif self.prompt_type == "dynamic_few_shot":

            if DYNAMIC_RETRIEVER is None:

                print("Loading Dynamic Few-Shot retriever...")

                DYNAMIC_RETRIEVER = DynamicFewShot()

            system_prompt = DYNAMIC_RETRIEVER.build_prompt(text)

        else:

            raise ValueError(
                f"Unknown prompt type: {self.prompt_type}"
            )

        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ]

    def predict(self, text: str):
        messages = self._build_prompt(text)

        # Model-specific settings
        if "qwen/qwen3.6" in self.model_name or "qwen/qwen3.8" in self.model_name:
            reasoning_effort = "none"
            max_tokens = 50
        
        elif "openai/gpt-oss" in self.model_name:
            reasoning_effort = "low"
            max_tokens = 300 if self.prompt_type == "few_shot" else 150

        else:
            reasoning_effort = None
            max_tokens = 50

        start = time.time()

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_completion_tokens": max_tokens
        }

        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        response = self.client.chat.completions.create(**kwargs)

        elapsed = time.time() - start

        prediction = response.choices[0].message.content

        if prediction is None:
            prediction = ""

        prediction = prediction.strip()

        return {
            "model": self.model_name,
            "prediction": prediction,
            "inference_time": f"{elapsed:.3f} seconds"
        }
    
    def load_model(self):
        print(f"Using Groq API model: {self.model_name}")
    
    def predict_batch(self, texts: list[str]):
        predictions = []
        for text in texts: 
            predictions.append(self.predict(text))
        
        return predictions;
    

    def get_model_name(self):

        return self.model_name