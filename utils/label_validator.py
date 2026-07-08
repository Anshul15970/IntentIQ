from datasets import load_dataset


class LabelValidator:

    def __init__(self):

        dataset = load_dataset(
            "PolyAI/banking77",
            trust_remote_code=True
        )

        self.valid_labels = set(
            dataset["train"].features["label"].names
        )

    def is_valid(self, prediction):

        return prediction in self.valid_labels

    def get_labels(self):

        return sorted(self.valid_labels)