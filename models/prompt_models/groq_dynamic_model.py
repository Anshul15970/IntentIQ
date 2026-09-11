from groq import Groq
from dotenv import load_dotenv
import os

from prompts.dynamic_few_shot import DynamicFewShot


load_dotenv()

_dynamic_few_shot = None

class GroqDynamicFewShot:

    def __init__(self, model_name):

        global _dynamic_few_shot
        
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.model_name = model_name
        
        if _dynamic_few_shot is None:
            print("Loading Dynamic Few-shot...")
            _dynamic_few_shot = DynamicFewShot()
        
        self.dynamic_few_shot = _dynamic_few_shot

    def predict(self, query):

        prompt = self.dynamic_few_shot.build_prompt(query)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_completion_tokens=20
        )

        prediction = response.choices[0].message.content.strip()

        print("GROQ RESPONSE:", repr(prediction))
        
        return {
            "prediction": prediction
        }