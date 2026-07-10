from models.hf_model import HFModel


class PhiModel(HFModel):

    def __init__(self, prompt_type="few_shot"):
        super().__init__(
            "microsoft/Phi-4-mini-instruct",
            prompt_type
        )