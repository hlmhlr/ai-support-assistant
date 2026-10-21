"""Connect LangChain to the local Ollama models.

Replaces the lab's `ModelInference` and `ChatWatsonx` setup. The models run
on local computer through Ollama instead of IBM's cloud.


Two ways to call a model:
- `generate_text`: plain text completion, like the lab's `model.generate(text)`.
  The prompt is sent exactly as written, with no chat template added.
- `get_chat_model`: a LangChain chat model, used from the LangChain stage on.


"""
from dataclasses import dataclass

import ollama

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


def model_tag(name: str) -> str:
    """Return the Ollama tag for "granite", "llama" or "mistral"."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose one of: {', '.join(MODELS)}")

    return MODELS[name]


@dataclass
class GenerationResult:
    """The answer from one call, plus the numbers used to compare models."""

    model: str            # short name, e.g. "granite"
    text: str             # the generated answer
    output_tokens: int    # how many tokens the model generated
    load_seconds: float   # time spent loading the model into memory
    total_seconds: float  # total time of the call, including loading

    @property
    def generation_seconds(self) -> float:
        """Time spent answering, without loading the model."""
        return self.total_seconds - self.load_seconds

    @property
    def tokens_per_second(self) -> float:
        """Generation speed. Measured locally instead of the lab's price per token."""
        if self.generation_seconds <= 0:
            return 0.0
        return self.output_tokens / self.generation_seconds


def generate_text(
    name: str,
    prompt: str,
    params: dict | None = None,
    client: ollama.Client | None = None,
) -> GenerationResult:
    """Send `prompt` to a model as plain text and return the answer with timings.

    `raw=True` tells Ollama not to wrap the prompt in the model's chat template,
    so the model sees exactly what watsonx's `generate()` would send it.
    `client` can be replaced by a fake one in tests.
    """
    client = client or ollama.Client(host=OLLAMA_BASE_URL)
    response = client.generate(
        model=model_tag(name),
        prompt=prompt,
        raw=True,
        options=to_ollama_options(params),
    )

    # Ollama reports durations in nanoseconds; divide by 1e9 to get seconds.
    return GenerationResult(
        model=name,
        text=response.response.strip(),
        output_tokens=response.eval_count or 0,
        load_seconds=(response.load_duration or 0) / 1e9,
        total_seconds=(response.total_duration or 0) / 1e9,
    )



    return ChatOllama(
        model=MODELS[name],
        base_url=OLLAMA_BASE_URL,
        **to_ollama_options(params),
    )
