from datasets import load_dataset


def load_banking77():

    dataset = load_dataset("PolyAI/banking77")

    return dataset


def get_label_names(dataset):

    return dataset["train"].features["label"].names


def format_example(example, label_names):

    return {
        "text": f"""<|im_start|>system
You are an intent classification assistant.

Return ONLY the intent label.

<|im_end|>

<|im_start|>user
{example["text"]}

<|im_end|>

<|im_start|>assistant
{label_names[example["label"]]}
<|im_end|>
"""
    }


def prepare_dataset(format_for_training=True):

    dataset = load_banking77()

    label_names = get_label_names(dataset)

    if format_for_training:

        train_dataset = dataset["train"].map(
            lambda x: format_example(x, label_names)
        )

        test_dataset = dataset["test"].map(
            lambda x: format_example(x, label_names)
        )

    else:

        train_dataset = dataset["train"]

        test_dataset = dataset["test"]

    return train_dataset, test_dataset, label_names