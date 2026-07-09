from models.hf_model import HFModel


class GemmaModel(HFModel):

    def __init__(self, prompt_type="few_shot"):
        super().__init__(
            "google/gemma-3-1b-it",
            prompt_type
        )