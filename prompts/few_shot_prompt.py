from datasets import load_dataset


def build_few_shot_prompt(num_examples=5):

    dataset = load_dataset(
        "PolyAI/banking77",
        trust_remote_code=True
    )

    train = dataset["train"]

    label_names = train.features["label"].names

    prompt = """
You are an intent classification expert.

Choose exactly ONE intent.

Return ONLY the intent label.

Examples:

"""

    used = set()

    for sample in train:

        intent = label_names[sample["label"]]

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