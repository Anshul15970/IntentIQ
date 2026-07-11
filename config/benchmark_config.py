from models.prompt_models.qwen_model import QwenModel

BENCHMARK_MODELS = [

    QwenModel("zero_shot"),
    QwenModel("few_shot"),
    QwenModel("dynamic_few_shot"),

]