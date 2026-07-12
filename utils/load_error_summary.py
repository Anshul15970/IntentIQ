import pandas as pd

def load_error_summary(model):

    mapping = {
        "Zero-shot":"results/zero_error_summary.csv",
        "Few-shot":"results/few_error_summary.csv",
        "Dynamic Few-shot":"results/dynamic_error_summary.csv",
        "LoRA":"results/lora_error_summary.csv"
    }

    return pd.read_csv(mapping[model]).head(20)