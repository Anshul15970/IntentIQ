from models.prompt_models.phi_model import PhiModel
import time

model = PhiModel()

print("Loading model...")

start = time.time()

model.load_model()

result = model.predict(
    "I lost my debit card yesterday."
)

end = time.time()

print(result)

print("Time: ", end - start)