from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from utils.label_validator import LabelValidator
from utils.intent_mapper import IntentMapper

class Benchmark:

    def __init__(self, model):
        self.model = model
        self.validator = LabelValidator()
        self.mapper = IntentMapper()

    def evaluate(self, texts, labels):

        predictions = []
        valid_predictions = 0
        invalid_predictions = 0
        
        for text in texts:
            result = self.model.predict(text)
            prediction = result["prediction"]
            if not self.validator.is_valid(prediction):
                prediction = self.mapper.map_prediction(prediction)
            predictions.append(prediction)
            if self.validator.is_valid(prediction):
                valid_predictions += 1
            else:
                invalid_predictions += 1

        accuracy = accuracy_score(labels, predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "valid_predictions": valid_predictions,
            "invalid_predictions": invalid_predictions,
            "predictions": predictions
        }