import os
import pandas as pd


class ResultLogger:

    def __init__(self, filepath="results/benchmark_results.csv"):
        self.filepath = filepath

    def log(self, model_name, num_samples, results):

        row = {
    "Model": model_name,
    "Samples": num_samples,

    "Accuracy": results["accuracy"],
    "Precision": results["precision"],
    "Recall": results["recall"],
    "F1": results["f1"],

    "Total Time (s)": round(results["total_time"], 2),
    "Avg Time / Query (s)": round(results["average_time"], 3),

    "Valid Predictions": results["valid_predictions"],
    "Invalid Predictions": results["invalid_predictions"]
}

        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            df = pd.read_csv(self.filepath)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_csv(self.filepath, index=False)

        print(f"Results saved to {self.filepath}")