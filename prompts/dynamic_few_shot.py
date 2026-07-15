import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch


class DynamicFewShot:

    def __init__(self):

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.train = pd.read_csv("data/banking77_train.csv")

        print("Encoding Banking77 training set...")

        self.train_embeddings = self.embedding_model.encode(
            self.train["text"].tolist(),
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
            k=20
        ).indices

        examples = []
        used_intents = set()

        for idx in top_indices:

            sample = self.train.iloc[idx.item()]

            intent = sample["intent"]

            if intent in used_intents:
                continue

            used_intents.add(intent)

            examples.append(
                {
                    "text": sample["text"],
                    "intent": intent
                }
            )

            if len(examples) == k:
                break

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