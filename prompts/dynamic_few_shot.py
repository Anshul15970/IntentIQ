from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch

class DynamicFewShot:

    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2")
        self.dataset = load_dataset(
            "PolyAI/banking77",
             trust_remote_code=True)

        self.train = self.dataset["train"]

        self.label_names = self.train.features["label"].names
        
        self.train = self.dataset["train"]

        self.label_names = self.train.features["label"].names

        print("Encoding Banking77 training set...")

        self.train_embeddings = self.embedding_model.encode(
            self.train["text"],
            convert_to_tensor=True,
            show_progress_bar=True
        )
    
    def retrieve_examples(self, query, k=5):

        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True
        )

        similarities = util.cos_sim(
            query_embedding,
            self.train_embeddings
        )[0]

        top_indices = torch.topk(
            similarities,
            k=k
        ).indices

        examples = []

        for idx in top_indices:

            sample = self.train[idx.item()]

            intent = self.label_names[sample["label"]]

            examples.append(
                {
                    "text": sample["text"],
                    "intent": intent
                }
            )

        return examples
    
    def build_prompt(self, query, k=5):

        examples = self.retrieve_examples(query, k)

        prompt = """
    You are an intent classification expert.

    Choose exactly ONE intent.

    Return ONLY the intent label.

    Examples:

    """

        for example in examples:

            prompt += f"""
    Sentence:
    {example['text']}

    Intent:
    {example['intent']}

    """

        prompt += f"""

    Now classify this sentence.

    Sentence:
    {query}

    Intent:
    """

        return prompt