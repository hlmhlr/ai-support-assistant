"""LangChain chains: a prompt template piped into a model, one per model family.

Replaces the lab's `model.py`. The structure is the same (initialize_model,
one PromptTemplate per model, get_ai_response with `template | model`), with
two changes:
- The templates come from templates.py, the formats checked in script 03.
- The model is OllamaLLM (text completion) in raw mode. The lab sent its
  special tokens to ChatWatsonx, a chat model, which wraps the prompt in its
  own template a second time. Raw mode sends the prompt exactly as written.
"""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaLLM

from support_assistant.config import OLLAMA_BASE_URL
from support_assistant.llm import model_tag, to_ollama_options
from support_assistant.templates import TEMPLATES


def initialize_model(name: str, params: dict | None = None) -> Runnable:
    """Create a text-completion model for "granite", "llama" or "mistral".

    `.bind(raw=True)` adds raw=True to every call, so Ollama does not add the
    model's chat template on top of ours.
    """
    llm = OllamaLLM(
        model=model_tag(name),
        base_url=OLLAMA_BASE_URL,
        **to_ollama_options(params),
    )
    return llm.bind(raw=True)


def build_template(name: str) -> PromptTemplate:
    """Turn a model's format from templates.py into a LangChain PromptTemplate."""
    model_tag(name)  # fail early with a clear message if the name is unknown
    return PromptTemplate(template=TEMPLATES[name], input_variables=["system", "user"])


def build_chain(name: str, llm: BaseLanguageModel | Runnable | None = None) -> Runnable:
    """Pipe the model's prompt template into the model: template | model.

    `llm` can be replaced by a fake model in tests.
    """
    return build_template(name) | (llm or initialize_model(name))


def get_ai_response(
    name: str,
    system_prompt: str,
    user_prompt: str,
    llm: BaseLanguageModel | Runnable | None = None,
) -> str:
    """Fill in the template, send it to the model and return the answer text.

    One function with a model name replaces the lab's llama_response,
    granite_response and mistral_response.
    """
    chain = build_chain(name, llm)
    answer = chain.invoke({"system": system_prompt, "user": user_prompt})
    return answer.strip()
