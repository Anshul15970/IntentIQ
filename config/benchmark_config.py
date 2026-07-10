from models.prompt_models.qwen_model import QwenModel
from models.prompt_models.gemma_model import GemmaModel

BENCHMARK_MODELS = [

    QwenModel("zero_shot"),
    QwenModel("few_shot"),
    QwenModel("dynamic_few_shot"),

    GemmaModel("zero_shot"),
    GemmaModel("few_shot"),
    GemmaModel("dynamic_few_shot"),

]