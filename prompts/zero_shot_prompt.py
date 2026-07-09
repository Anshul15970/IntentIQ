def build_zero_shot_prompt():

    prompt = """
You are an intent classification expert.

Choose exactly ONE intent.

Return ONLY the intent label.

Now classify the following sentence.
"""

    return prompt