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
        
    def plot_f1(self):

        plt.figure(figsize=(8, 5))

        labels = self.df["Model"] + "\n(" + self.df["Prompt Type"] + ")"

        plt.bar(
            labels,
            self.df["F1"]
        )

        plt.title("F1 Score Comparison")

        plt.xlabel("Model")

        plt.ylabel("F1 Score")

        plt.ylim(0, 1)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "f1_comparison.png"
            )
        )

        plt.close()

        print("Saved F1 graph.")
        
    def plot_time(self):

        plt.figure(figsize=(8,5))

        labels = self.df["Model"] + "\n(" + self.df["Prompt Type"] + ")"

        plt.bar(
            labels,
            self.df["Avg Time / Query (s)"]
        )

        plt.title("Average Inference Time")

        plt.xlabel("Model")

        plt.ylabel("Seconds")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "time_comparison.png"
            )
        )

        plt.close()

        print("Saved inference time graph.")
        
    def plot_tradeoff(self):

        plt.figure(figsize=(7,6))

        plt.scatter(
            self.df["Avg Time / Query (s)"],
            self.df["Accuracy"]
        )

        for _, row in self.df.iterrows():

            plt.text(
                row["Avg Time / Query (s)"],
                row["Accuracy"],
                row["Model"] + "\n" + row["Prompt Type"],
                fontsize=8
            )

        plt.xlabel("Avg Time (s)")

        plt.ylabel("Accuracy")

        plt.title("Accuracy vs Inference Time")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "accuracy_vs_speed.png"
            )
        )

        plt.close()

        print("Saved tradeoff graph.")