"""Tests for model.py: the template | model chains.

LangChain's FakeListLLM returns prepared answers, and a RunnableLambda
records the exact prompt the model would receive.
"""

import json

import pytest
from langchain_core.language_models import FakeListLLM
from langchain_core.runnables import RunnableLambda

from support_assistant.config import MODELS
from support_assistant.model import (
    StructuredOutputError,
    get_ai_response,
    get_structured_response,
    initialize_model,
)


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


# --- Structured output: template | model | json_parser, checked by AIResponse ---

GOOD_JSON = json.dumps(
    {
        "summary": "Package marked delivered but not received.",
        "sentiment": 25,
        "response": "I'm sorry your package hasn't arrived. We'll look into it.",
        "next_action": "Open a carrier trace for the tracking number.",
    }
)


def test_valid_json_becomes_an_ai_response():
    fake = FakeListLLM(responses=[f"```json\n{GOOD_JSON}\n```"])  # code fences are allowed
    answer = get_structured_response("granite", "Where is my parcel?", llm=fake)
    assert answer.sentiment == 25
    assert answer.next_action.startswith("Open a carrier trace")


def test_format_instructions_go_into_the_system_turn():
    prompts = []
    recorder = RunnableLambda(lambda prompt: prompts.append(prompt.to_string()) or GOOD_JSON)

    get_structured_response("mistral", "Where is my parcel?", llm=recorder)

    instructions, user_text = prompts[0].split("\n\nWhere is my parcel?")
    assert '"next_action"' in instructions  # the schema is described before the message
    assert user_text == "[/INST]"


def test_invalid_answer_is_retried_with_the_reason():
    prompts = []
    answers = iter(['{"summary": "x", "sentiment": 250}', GOOD_JSON])
    recorder = RunnableLambda(lambda prompt: prompts.append(prompt.to_string()) or next(answers))

    answer = get_structured_response("llama", "Where is my parcel?", llm=recorder)

    assert answer.sentiment == 25  # the second answer was used
    assert len(prompts) == 2
    assert "sentiment: Input should be less than or equal to 100" in prompts[1]
    assert "next_action: Field required" in prompts[1]


def test_truncated_json_is_not_accepted_as_empty():
    # The parser turns '{"summary": ' into {} without an error; the schema catches it.
    fake = FakeListLLM(responses=['{"summary": ', '{"summary": '])
    with pytest.raises(StructuredOutputError, match="summary: Field required"):
        get_structured_response("granite", "Hi", llm=fake)


def test_error_after_two_bad_answers_keeps_the_raw_output():
    fake = FakeListLLM(responses=["I can't help with that.", "Still no JSON."])
    with pytest.raises(StructuredOutputError) as caught:
        get_structured_response("granite", "Hi", llm=fake)

    assert caught.value.problem == "the answer was not valid JSON"
    assert caught.value.raw_output == "Still no JSON."
