from .openai_handler import OpenAIHandler
from .huggingface_handler import HuggingFaceHandler
from .model_handler_base import ModelHandlerBase
from .anthropic_handler import Claude35SonnetHandler
from .gemini_handler import GeminiHandler
from .deepseek_handler import DeepSeekHandler 
from .medgemma import medgemma 

import os  # To access environment variables

def load_model_handler(config):
    """
    Factory method to load the appropriate model handler based on config.

    """
    model_type = config["model"]["type"]

    if model_type == "openai":
        # retrieve API key from environment
        api_key = config["model"].get("api_key", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError(
                "API key is not provided in config or environment variable (OPENAI_API_KEY)."
            )
        return OpenAIHandler(
            api_key=api_key,
            model=config["model"]["name"]
        )
    elif model_type == "huggingface":
        return HuggingFaceHandler(
            model_name=config["model"]["name"],
            cache_dir=config.get("cache_dir", None)
            #device=config["model"].get("device", "cpu")
        )

    elif model_type == "medgemma":
        return medgemma(
            model_name=config["model"]["name"],
            cache_dir=config.get("cache_dir", None)
            #device=config["model"].get("device", "cpu")
        )

    elif model_type == "gemini":
        # retrieve API key from environment
        api_key = config["model"].get("api_key", os.getenv("GEMINI_API_KEY"))
        if not api_key:
            raise ValueError(
                "API key is not provided in config or environment variable (OPENAI_API_KEY)."
            )
        return GeminiHandler(
            api_key=api_key,
            model=config["model"]["name"]
        )

    elif model_type == "deepseek":  # 
        api_key = config["model"].get("api_key", os.getenv("DEEPSEEK_API_KEY"))
        if not api_key:
            raise ValueError("API key is not provided for DeepSeek.")
        return DeepSeekHandler(api_key=api_key, model=config["model"]["name"])

    elif model_type == "anthropic":
        # Retrieve Anthropic API key from environment or config
        api_key = config["model"].get("api_key", os.getenv("ANTHROPIC_API_KEY"))
        if not api_key:
            raise ValueError(
                "API key is not provided in config or environment variable (ANTHROPIC_API_KEY)."
            )
        return Claude35SonnetHandler(
            api_key=api_key,
            model_name=config["model"].get("name", "claude-3-5-sonnet-20240620")
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
