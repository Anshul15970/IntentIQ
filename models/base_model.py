from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all intent classification models.
    Every model (Qwen, Gemma, Phi, etc.) must inherit from this class.
    """

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def predict(self, text: str):
        """
        Predict the intent of a single sentence.

        Parameters:
            text (str): Input sentence.

        Returns:
            dict: Prediction result.
        """
        pass

    @abstractmethod
    def get_model_name(self):
        """Return the model name."""
        pass