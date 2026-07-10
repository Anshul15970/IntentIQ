from prompts.dynamic_few_shot import DynamicFewShot

retriever = DynamicFewShot()

prompt = retriever.build_prompt(
    "I lost my debit card yesterday."
)

print(prompt)