from models.prompt_models.gemma_model import GemmaModel

model = GemmaModel()

print("Loading model...")

model.load_model()

result = model.predict(
    "I lost my debit card yesterday."
)

print(result)