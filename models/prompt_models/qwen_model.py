from models.hf_model import HFModel


class QwenModel(HFModel):

    def __init__(self, prompt_type="few_shot"):
        super().__init__(
            "Qwen/Qwen2.5-0.5B-Instruct",
            prompt_type
        )