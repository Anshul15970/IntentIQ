from datasets import load_dataset

from models.prompt_models.qwen_model import QwenModel
from benchmark.benchmark import Benchmark

# Load dataset
dataset = load_dataset(
    "PolyAI/banking77",
    trust_remote_code=True
)

# Get label names
label_names = dataset["test"].features["label"].names

# Use first 100 test samples
test_data = dataset["test"].select(range(100))

texts = test_data["text"]
labels = [label_names[i] for i in test_data["label"]]

# Load model
model = QwenModel()
model.load_model()

# Run benchmark
benchmark = Benchmark(model)

results = benchmark.evaluate(texts, labels)

print(results)