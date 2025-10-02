import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain.chat_models import init_chat_model


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


def get_current_dir() -> Path:
    """Get the current directory of the module.
    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()


def create_model(model_name: str, temperature: float = 0.0, max_tokens: Optional[int] = None, **kwargs):
    """Create a model instance based on the model name with appropriate configuration.

    Args:
        model_name: The model identifier (e.g., "ibm:meta-llama/llama-3-1-70b-instruct",
                   "openai:gpt-4", "anthropic:claude-sonnet-4", "ollama:llama3")
        temperature: Temperature for model generation (default: 0.0)
        max_tokens: Maximum tokens to generate (optional)
        **kwargs: Additional parameters to pass to init_chat_model

    Returns:
        Initialized chat model instance

    Raises:
        ValueError: If the model type is not supported
    """
    if model_name.startswith("ibm:"):
        from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

        params = TextChatParameters(max_tokens=max_tokens or 500, temperature=temperature, seed=122)

        wx_credentials = {
            "url": os.getenv("IBM_CLOUD_URL"),
            "apikey": os.getenv("WATSONX_APIKEY"),
            "project_id": os.getenv("WATSONX_PROJECT_ID"),
        }

        return init_chat_model(model=model_name, params=params, **wx_credentials, **kwargs)

    elif model_name.startswith(("openai:", "anthropic:")):
        model_kwargs = {"temperature": temperature}
        if max_tokens:
            model_kwargs["max_tokens"] = max_tokens
        model_kwargs.update(kwargs)

        return init_chat_model(model=model_name, **model_kwargs)

    elif model_name.startswith("ollama:"):
        # Ollama models - special handling with num_predict
        model_kwargs = {"temperature": temperature, "seed": 122, "num_predict": max_tokens or 1000}
        model_kwargs.update(kwargs)

        return init_chat_model(model=model_name, **model_kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_name}. Supported types: 'ibm:', 'openai:', 'anthropic:', 'ollama:'")
