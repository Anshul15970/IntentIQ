from models.hf_model import HFModel


class LlamaModel(HFModel):

    def __init__(self, prompt_type="few_shot"):
        super().__init__(
            "meta-llama/Llama-3.2-1B-Instruct",
            prompt_type
        )