"""Tests for schemas.py: which answers AIResponse accepts and rejects."""

import pytest
from pydantic import ValidationError

from support_assistant.schemas import AIResponse

VALID = {
    "summary": "Customer was charged twice for order #48213.",
    "sentiment": 10,
    "response": "I'm sorry about the double charge. Our billing team is looking into it.",
    "next_action": "Check order #48213 for a duplicate payment and refund it.",
}


def test_valid_answer_is_accepted():
    assert AIResponse.model_validate(VALID).sentiment == 10


def test_spaces_around_text_are_removed():
    answer = AIResponse.model_validate({**VALID, "summary": "  Double charge.  "})
    assert answer.summary == "Double charge."


@pytest.mark.parametrize("sentiment", [-1, 101, 250])
def test_sentiment_must_be_between_0_and_100(sentiment):
    with pytest.raises(ValidationError):
        AIResponse.model_validate({**VALID, "sentiment": sentiment})


def test_missing_next_action_is_rejected():
    answer = {key: value for key, value in VALID.items() if key != "next_action"}
    with pytest.raises(ValidationError):
        AIResponse.model_validate(answer)


def test_empty_reply_is_rejected():
    with pytest.raises(ValidationError):
        AIResponse.model_validate({**VALID, "response": "   "})
