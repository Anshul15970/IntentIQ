import os
import pandas as pd
import matplotlib.pyplot as plt


class BenchmarkPlotter:

    def __init__(self,
                 csv_path="results/benchmark_results.csv",
                 output_dir="results"):

        self.csv_path = csv_path
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.df = pd.read_csv(csv_path)

    def plot_accuracy(self):

        plt.figure(figsize=(8, 5))

        labels = self.df["Model"] + "\n(" + self.df["Prompt Type"] + ")"

        plt.bar(
            labels,
            self.df["Accuracy"]
        )

        plt.title("Accuracy Comparison Across Models and Prompt Types")

        plt.xlabel("Model")

        plt.ylabel("Accuracy")

        plt.ylim(0, 1)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "accuracy_comparison.png"
            )
        )

        plt.close()

        print("Saved accuracy graph.")