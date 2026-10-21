"""Tests for llm.py: parameter translation, model names and timing numbers.

A fake Ollama client stands in for the real one, so no model is needed.
"""

import pytest
from ollama import ChatResponse, GenerateResponse, Message

from support_assistant.llm import (
    GenerationResult,
    chat_text,
    generate_text,
    model_tag,
    to_ollama_options,
)

# Durations as Ollama reports them, in nanoseconds.
TIMINGS = dict(
    eval_count=10,
    load_duration=2_000_000_000,   # 2 s loading
    total_duration=3_000_000_000,  # 3 s in total
    eval_duration=500_000_000,     # 0.5 s writing the 10 tokens
)


class FakeClient:
    """Records the last call and returns a fixed answer."""

    def __init__(self):
        self.last_call = None

    def generate(self, **kwargs):
        self.last_call = kwargs
        return GenerateResponse(model=kwargs["model"], response="  Ottawa.\n", **TIMINGS)

    def chat(self, **kwargs):
        self.last_call = kwargs
        return ChatResponse(
            model=kwargs["model"],
            message=Message(role="assistant", content=" Billing "),
            **TIMINGS,
        )


def test_greedy_decoding_means_temperature_zero():
    assert to_ollama_options({"decoding_method": "greedy", "max_new_tokens": 100}) == {
        "num_predict": 100,
        "temperature": 0.0,
    }


def test_sampling_keeps_the_temperature():
    options = to_ollama_options({"decoding_method": "sample", "temperature": 0.9})
    assert options["temperature"] == 0.9


def test_unknown_model_name_lists_the_valid_ones():
    with pytest.raises(ValueError, match="granite, llama, mistral"):
        model_tag("gpt")


def test_generate_text_sends_the_prompt_raw():
    client = FakeClient()
    result = generate_text("granite", "What is the capital of Canada?", client=client)

    assert client.last_call["raw"] is True
    assert client.last_call["prompt"] == "What is the capital of Canada?"
    assert result.text == "Ottawa."  # surrounding spaces and newlines removed


def test_chat_text_sends_system_and_user_messages():
    client = FakeClient()
    result = chat_text("llama", "Be brief.", "Which team?", client=client)

    roles = [message["role"] for message in client.last_call["messages"]]
    assert roles == ["system", "user"]
    assert result.text == "Billing"


def test_timings_are_converted_to_seconds():
    result = generate_text("mistral", "hi", client=FakeClient())

    assert result.load_seconds == 2.0
    assert result.generation_seconds == 1.0  # total minus loading
    assert result.tokens_per_second == 20.0  # 10 tokens / 0.5 s of writing


def test_tokens_per_second_is_zero_without_timing():
    result = GenerationResult("granite", "x", 3, 0.0, 0.0, 0.0)
    assert result.tokens_per_second == 0.0
