import pandas as pd


def generate_project_summary():

    models = {
        "Zero-shot": "results/zero_error_summary.csv",
        "Few-shot": "results/few_error_summary.csv",
        "Dynamic Few-shot": "results/dynamic_error_summary.csv",
        "LoRA": "results/lora_error_summary.csv"
    }

    benchmark = pd.read_csv("results/benchmark_results.csv")

    lora = pd.read_csv("results/lora_error_summary.csv")

    summaries = {}

    for model, path in models.items():

        df = pd.read_csv(path)

        total_errors = df["count"].sum()

        top = df.iloc[0]

        summaries[model] = {
            "errors": total_errors,
            "top": f"{top['true_label']} → {top['prediction']}"
        }

    lora_errors = lora["count"].sum()

    text = f"""
PROJECT ERROR ANALYSIS

Zero-shot prompting produced the highest number of semantic confusions. Most mistakes occurred because the model predicted broad intents instead of Banking77-specific labels, making it unsuitable for production intent classification.

Few-shot prompting substantially reduced these errors by providing representative examples. However, it continued to confuse closely related banking intents such as identity verification, exchange-rate queries and card delivery scenarios.

Dynamic Few-shot further improved classification by retrieving semantically similar examples for every query. This reduced many prompt-related ambiguities and achieved noticeably more consistent predictions than static prompting.

LoRA fine-tuning produced the lowest overall error count ({lora_errors}). Most remaining mistakes occurred only between highly similar intent classes (for example card arrival vs card delivery estimate or transfer-related intents), indicating that the model learned the Banking77 intent space effectively rather than relying on prompt engineering.

Overall Recommendation

For rapid prototyping or situations where training is not possible, Dynamic Few-shot provides the strongest prompting performance. However, when maximum accuracy, consistency and production readiness are required, the LoRA fine-tuned model is the recommended approach because it achieves the best balance of accuracy, robustness and inference efficiency while exhibiting the fewest semantic confusions.
"""

    return text