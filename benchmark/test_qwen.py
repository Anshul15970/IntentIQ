from models.prompt_models.qwen_model import QwenModel

model = QwenModel()

print("Loading model...")
model.load_model()

result = model.predict("I lost my debit card yesterday.")

print(result)