from models.prompt_models.qwen_model import QwenModel

m1 = QwenModel(prompt_type="zero_shot")
m2 = QwenModel(prompt_type="few_shot")

print(m1.prompt_type)
print(m2.prompt_type)