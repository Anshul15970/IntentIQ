import pandas as pd

MODEL = "dynamic"     # change to zero, dynamic, or lora when needed

df = pd.read_csv(f"results/{MODEL}_errors.csv")

errors = df[df["correct"] == False]

errors.to_csv(
    f"results/{MODEL}_errors.csv",
    index=False
)

summary = (
    errors
    .groupby(["true_label", "prediction"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

summary.to_csv(
    f"results/{MODEL}_error_summary.csv",
    index=False
)

print(summary.head(20))