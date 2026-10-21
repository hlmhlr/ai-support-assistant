"""Tests for model.py: the template | model chains.

LangChain's FakeListLLM returns prepared answers, and a RunnableLambda
records the exact prompt the model would receive.
"""

from langchain_core.language_models import FakeListLLM
from langchain_core.runnables import RunnableLambda

from support_assistant.config import MODELS
from support_assistant.model import get_ai_response, initialize_model


def test_chain_returns_the_model_answer_without_extra_spaces():
    fake = FakeListLLM(responses=["  The Billing team will refund you.\n"])
    answer = get_ai_response("granite", "Be kind.", "I was charged twice.", llm=fake)
    assert answer == "The Billing team will refund you."


def test_chain_sends_the_model_specific_prompt():
    prompts = []
    recorder = RunnableLambda(lambda prompt: prompts.append(prompt.to_string()) or "ok")

    get_ai_response("mistral", "Be kind.", "Where is my parcel?", llm=recorder)

    assert prompts == ["[INST] Be kind.\n\nWhere is my parcel?[/INST]"]


def test_model_is_called_in_raw_mode_with_greedy_decoding():
    model = initialize_model("llama", {"max_new_tokens": 100})

    assert model.kwargs == {"raw": True}  # added to every call
    assert model.bound.model == MODELS["llama"]  # the tag set in config or .env
    assert model.bound.temperature == 0.0
    assert model.bound.num_predict == 100
