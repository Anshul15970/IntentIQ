from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sentence_transformers import util
class IntentMapper: 
    def __init__(self):

        dataset = load_dataset(
            "PolyAI/banking77",
            trust_remote_code=True
        )

        self.labels = dataset["train"].features["label"].names
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.label_embeddings = self.model.encode(self.labels,convert_to_tensor=True)

    def map_prediction(self, prediction):

        prediction_embedding = self.model.encode(
            prediction,
            convert_to_tensor=True
        )

        similarities = util.cos_sim(
            prediction_embedding,
            self.label_embeddings
        )[0]

        best_index = similarities.argmax().item()

        return self.labels[best_index]