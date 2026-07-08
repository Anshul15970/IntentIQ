from models.hf_model import HuggingFaceModel


class QwenModel(HuggingFaceModel):

    def __init__(self):

        super().__init__(
            "Qwen/Qwen2.5-0.5B-Instruct"
        )