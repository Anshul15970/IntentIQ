import pandas as pd

print("Building few-shot prompt...")

train = pd.read_csv("data/banking77_train.csv")


def build_few_shot_prompt(num_examples=5):

    prompt = """
You are an intent classification expert.

Choose exactly ONE intent.

Return ONLY the intent label.

Examples:

"""

    used = set()

    for _, sample in train.iterrows():

        intent = sample["intent"]

        if intent in used:
            continue

        prompt += f"""
Sentence:
{sample['text']}

Intent:
{intent}

"""

        used.add(intent)

        if len(used) == num_examples:
            break

    prompt += "\nNow classify the following sentence.\n"

    return prompt