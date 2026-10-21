"""Tests for templates.py: each model gets its own special tokens."""

import pytest

from support_assistant.templates import format_prompt


def test_granite_uses_role_tokens_and_opens_the_assistant_turn():
    prompt = format_prompt("granite", "Be brief.", "Hi")
    assert prompt.startswith("<|start_of_role|>system<|end_of_role|>Be brief.<|end_of_text|>")
    assert prompt.endswith("<|start_of_role|>assistant<|end_of_role|>")


def test_llama_uses_llama_3_tokens_not_llama_4():
    prompt = format_prompt("llama", "Be brief.", "Hi")
    assert "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>" in prompt
    assert "<|header_start|>" not in prompt  # Llama 4 token used in the lab


def test_mistral_puts_the_system_text_inside_the_instruction():
    assert format_prompt("mistral", "Be brief.", "Hi") == "[INST] Be brief.\n\nHi[/INST]"


@pytest.mark.parametrize("name", ["granite", "llama", "mistral"])
def test_no_hand_written_start_token(name):
    # Ollama's tokenizer adds the start token; writing it too would double it.
    prompt = format_prompt(name, "s", "u")
    assert not prompt.startswith(("<s>", "<|begin_of_text|>"))


def test_braces_in_customer_text_are_kept():
    prompt = format_prompt("mistral", "s", "Order {A-17} is late")
    assert "Order {A-17} is late" in prompt


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        format_prompt("gpt", "s", "u")
