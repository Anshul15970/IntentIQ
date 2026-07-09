from datasets import load_dataset

from models.prompt_models.qwen_model import QwenModel
from models.prompt_models.gemma_model import GemmaModel
from benchmark.benchmark import Benchmark
from utils.result_logger import ResultLogger
from visualization.plot_results import BenchmarkPlotter
# Load dataset
dataset = load_dataset(
    "PolyAI/banking77",
    trust_remote_code=True
)

# Get label names
label_names = dataset["test"].features["label"].names

# Use first 100 test samples
NUM_SAMPLES = 50
test_data = dataset["test"].select(range(NUM_SAMPLES))

texts = test_data["text"]
labels = [label_names[i] for i in test_data["label"]]

# Load model
models = [
    QwenModel(prompt_type="zero_shot"),
    QwenModel(prompt_type="few_shot"),
    GemmaModel(prompt_type="zero_shot"),
    GemmaModel(prompt_type="few_shot")
]
logger = ResultLogger()
# Run benchmark
for model in models:
    print(f"\nRunning benchmark for {model.get_model_name()}")
    model.load_model()
    benchmark = Benchmark(model)
    results = benchmark.evaluate(texts, labels)
    print(results)
    logger.log(
    model.get_model_name(),
    model.prompt_type,
    NUM_SAMPLES,
    results
)
plotter = BenchmarkPlotter()
plotter.plot_accuracy()