from datasets import load_dataset

from benchmark.benchmark import Benchmark
from utils.result_logger import ResultLogger
from visualization.plot_results import BenchmarkPlotter
from config.benchmark_config import BENCHMARK_MODELS

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import os

# -----------------------------
# Create results folder
# -----------------------------
os.makedirs("results", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("PolyAI/banking77")

label_names = dataset["test"].features["label"].names

NUM_SAMPLES = len(dataset["test"])

test_data = dataset["test"]

texts = test_data["text"]
labels = [label_names[i] for i in test_data["label"]]

logger = ResultLogger()

# -----------------------------
# Run benchmark
# -----------------------------
for model in BENCHMARK_MODELS:

    print(
        f"\nRunning benchmark for "
        f"{model.get_model_name()} "
        f"({model.prompt_type})"
    )

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

    # -----------------------------
    # Save predictions
    # -----------------------------
    pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "prediction": results["predictions"],
        "correct": [
            t == p
            for t, p in zip(labels, results["predictions"])
        ]
    }).to_csv(
        f"results/{model.prompt_type}_predictions.csv",
        index=False
    )

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(
        labels,
        results["predictions"],
        labels=label_names
    )

    plt.figure(figsize=(22, 22))

    plt.imshow(cm, interpolation="nearest")

    plt.title(
        f"{model.get_model_name()} ({model.prompt_type})"
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(
        range(len(label_names)),
        label_names,
        rotation=90,
        fontsize=6
    )

    plt.yticks(
        range(len(label_names)),
        label_names,
        fontsize=6
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        f"results/{model.prompt_type}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

# -----------------------------
# Plots
# -----------------------------
plotter = BenchmarkPlotter()

plotter.plot_accuracy()
plotter.plot_f1()
plotter.plot_time()
plotter.plot_tradeoff()