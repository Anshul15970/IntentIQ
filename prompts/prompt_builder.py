from datasets import load_dataset


def get_intent_labels():

    dataset = load_dataset(
    "PolyAI/banking77",
    "default"
)

    return dataset["train"].features["label"].names


def build_prompt():

    labels = get_intent_labels()

    label_text = "\n".join(f"- {label}" for label in labels)

    prompt = f"""
You are an intent classification model.

Your task is to classify the user's query into exactly one of the following Banking77 intents.

Intent Labels:

{label_text}

Rules:
1. Return ONLY one intent label.
2. Do not explain.
3. Do not write a sentence.
4. Return exactly one label.
"""

    return prompt