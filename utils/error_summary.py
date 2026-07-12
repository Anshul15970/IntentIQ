import pandas as pd


def summarize_errors(model):

    mapping = {
        "Zero-shot": "results/zero_error_summary.csv",
        "Few-shot": "results/few_error_summary.csv",
        "Dynamic Few-shot": "results/dynamic_error_summary.csv",
        "LoRA": "results/lora_error_summary.csv"
    }

    df = pd.read_csv(mapping[model])

    total_errors = df["count"].sum()

    top = df.iloc[0]

    summary = f"""
Model: {model}

Total misclassifications: {total_errors}

Most common confusion:
{top['true_label']} → {top['prediction']}

Occurrences: {top['count']}

Top 5 confusions:

"""

    for _, row in df.head(5).iterrows():
        summary += (
            f"• {row['true_label']} → "
            f"{row['prediction']} "
            f"({row['count']})\n"
        )

    return summary