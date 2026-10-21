"""Connect LangChain to the local Ollama models.

Replaces the lab's `ModelInference` and `ChatWatsonx` setup. The models run
on your own computer through Ollama instead of IBM's cloud.
"""

from langchain_ollama import ChatOllama

from support_assistant.config import MODELS, OLLAMA_BASE_URL, PARAMETERS


def to_ollama_options(params: dict | None = None) -> dict:
    """Translate the lab's parameter names into Ollama's names.

    Any values in `params` override the defaults, e.g. {"max_new_tokens": 100}.
    """
    settings = {**PARAMETERS, **(params or {})}

    if settings["decoding_method"] == "greedy":
        # Temperature 0 makes Ollama pick the most likely token every time,
        # which is what watsonx calls greedy decoding.
        temperature = 0.0
    else:
        temperature = settings["temperature"]

    return {
        "num_predict": settings["max_new_tokens"],  # Ollama's name for max_new_tokens
        "temperature": temperature,
    }


def get_chat_model(name: str, params: dict | None = None) -> ChatOllama:
    """Create a chat model for one of the keys in MODELS ("granite", "llama", "mistral")."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose one of: {', '.join(MODELS)}")

    return ChatOllama(
        model=MODELS[name],
        base_url=OLLAMA_BASE_URL,
        **to_ollama_options(params),
    )
