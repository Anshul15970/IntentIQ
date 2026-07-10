from models.prompt_models.llama_model import LlamaModel

TEST_SENTENCE = "I lost my debit card yesterday."

if __name__ == "__main__":

    print("Loading model...")

    model = LlamaModel(prompt_type="few_shot")
    model.load_model()

    result = model.predict(TEST_SENTENCE)

    print(result)