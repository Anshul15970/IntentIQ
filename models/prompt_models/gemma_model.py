from models.hf_model import HuggingFaceModel


class GemmaModel(HuggingFaceModel):

    def __init__(self):

        super().__init__(
            "google/gemma-3-1b-it"
        )