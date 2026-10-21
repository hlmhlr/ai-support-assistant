"""Tests for app.py: the page, the /generate API and its status codes.

Flask's test client sends requests without starting a server, and a fake
responder stands in for the model, so no Ollama is needed.
"""

import httpx
import ollama
import pytest

from app import MAX_MESSAGE_LENGTH, create_app
from support_assistant.model import StructuredOutputError
from support_assistant.schemas import AIResponse

ANSWER = AIResponse(
    summary="Customer was charged twice.",
    sentiment=15,
    response="Sorry about the double charge. Our billing team will review it.",
    next_action="Check the order for a duplicate payment.",
)


def client_with(responder):
    """A test client for an app whose model is replaced by `responder`."""
    return create_app(responder).test_client()


def raising(exception):
    """A fake responder that raises `exception` instead of answering."""

    def responder(model, message):
        raise exception

    return responder


def test_page_lists_every_model():
    page = client_with(lambda model, message: ANSWER).get("/")
    assert page.status_code == 200
    for name in ("Granite", "Llama", "Mistral"):
        assert name in page.text


def test_generate_returns_all_fields_with_model_and_duration():
    calls = []

    def responder(model, message):
        calls.append((model, message))
        return ANSWER

    reply = client_with(responder).post(
        "/generate", json={"message": "  I was charged twice.  ", "model": "llama"}
    )

    assert reply.status_code == 200
    body = reply.get_json()
    assert body["sentiment"] == 15
    assert body["next_action"] == "Check the order for a duplicate payment."
    assert body["model"] == "llama"
    assert body["duration"] >= 0
    assert calls == [("llama", "I was charged twice.")]  # message trimmed


@pytest.mark.parametrize(
    "body",
    [
        {"model": "granite"},                                  # no message
        {"message": "   ", "model": "granite"},                # empty message
        {"message": 42, "model": "granite"},                   # not text
        {"message": "Hi"},                                     # no model
        {"message": "Hi", "model": "gpt"},                     # unknown model
        {"message": "x" * (MAX_MESSAGE_LENGTH + 1), "model": "granite"},  # too long
        ["not", "an", "object"],                               # wrong JSON type
    ],
)
def test_bad_requests_get_400_and_never_reach_the_model(body):
    def responder(model, message):
        raise AssertionError("the model should not be called")

    reply = client_with(responder).post("/generate", json=body)

    assert reply.status_code == 400
    assert "error" in reply.get_json()


def test_body_that_is_not_json_gets_400():
    reply = client_with(lambda model, message: ANSWER).post("/generate", data="hello")
    assert reply.status_code == 400


def test_unusable_model_answer_gets_422_with_the_problem():
    error = StructuredOutputError("mistral", "sentiment: must be <= 100", "{}")
    reply = client_with(raising(error)).post(
        "/generate", json={"message": "Hi", "model": "mistral"}
    )

    assert reply.status_code == 422
    assert reply.get_json()["problem"] == "sentiment: must be <= 100"


@pytest.mark.parametrize(
    "exception",
    [httpx.ConnectError("Connection refused"), ConnectionError("Failed to connect")],
)
def test_ollama_not_running_gets_503(exception):
    reply = client_with(raising(exception)).post(
        "/generate", json={"message": "Hi", "model": "granite"}
    )

    assert reply.status_code == 503
    assert "Ollama" in reply.get_json()["error"]


def test_model_not_downloaded_gets_503_with_the_pull_command():
    exception = ollama.ResponseError('model "granite4:tiny-h" not found', 404)
    reply = client_with(raising(exception)).post(
        "/generate", json={"message": "Hi", "model": "granite"}
    )

    assert reply.status_code == 503
    assert "ollama pull" in reply.get_json()["error"]


def test_unexpected_error_gets_500_without_internal_details():
    exception = RuntimeError("secret internal detail")
    reply = client_with(raising(exception)).post(
        "/generate", json={"message": "Hi", "model": "granite"}
    )

    assert reply.status_code == 500
    assert "secret internal detail" not in reply.text
