class ModelHandlerBase:
    """
    Abstract base class for model handlers.
    All model handlers must implement the `prompt` method.
    """
    def prompt(self, input_text, **kwargs):
        """
        Generate a response from the model.

        """
        raise NotImplementedError("Subclasses must implement the `prompt` method.")
