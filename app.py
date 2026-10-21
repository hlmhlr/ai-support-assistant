"""JSON API for the support assistant.

Replaces the lab's `app.py`. POST /generate takes a customer message and a
model name and returns the summary, sentiment, reply and next action.

Changes from the lab:
- One status code per kind of problem, so the caller knows what went wrong:
  400 bad request, 422 unusable model answer, 503 Ollama or model not
  available, 500 anything else. The lab returned 500 with the raw error text.
- Unexpected errors are logged on the server, not sent to the browser.
- create_app() takes the function that produces the answer, so tests can
  pass a fake one and run without Ollama.

Run it with:  uv run python app.py   (then open http://127.0.0.1:5000)
"""

import time

import httpx
import ollama
from flask import Flask, jsonify, request

from support_assistant.config import MODELS
from support_assistant.model import StructuredOutputError, get_structured_response

# Longer messages are rejected: they slow the model down and a support
# message rarely needs more.
MAX_MESSAGE_LENGTH = 4000


def error(message: str, status: int, **details):
    """A JSON error body with an HTTP status code."""
    return jsonify({"error": message, **details}), status


def create_app(responder=get_structured_response) -> Flask:
    """Build the Flask app. `responder(model, message)` must return an AIResponse."""
    app = Flask(__name__)

    @app.post("/generate")
    def generate():
        # silent=True returns None instead of raising when the body isn't JSON.
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return error("Send a JSON body with 'message' and 'model'.", 400)

        message = data.get("message")
        model = data.get("model")
        if not isinstance(message, str) or not message.strip() or not model:
            return error("Missing message or model selection.", 400)
        if model not in MODELS:
            return error(f"Unknown model {model!r}. Choose one of: {', '.join(MODELS)}.", 400)
        if len(message) > MAX_MESSAGE_LENGTH:
            return error(f"Message is longer than {MAX_MESSAGE_LENGTH} characters.", 400)

        start = time.perf_counter()
        try:
            result = responder(model, message.strip())
        except StructuredOutputError as exc:
            # The model answered, but not in a usable form, even after a retry.
            return error(str(exc), 422, problem=exc.problem)
        except (httpx.TransportError, ConnectionError):
            # LangChain raises httpx errors, the plain ollama client ConnectionError.
            return error("Can't reach Ollama. Start the Ollama app and try again.", 503)
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                return error(f"Model not downloaded. Run: ollama pull {MODELS[model]}", 503)
            app.logger.exception("Ollama returned an error")
            return error("Ollama returned an error. See the server log.", 503)
        except Exception:
            app.logger.exception("Unexpected error in /generate")
            return error("Something went wrong. See the server log.", 500)

        return jsonify(
            {
                **result.model_dump(),
                "model": model,
                "duration": round(time.perf_counter() - start, 2),  # seconds, as in the lab
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    # Local use only: 127.0.0.1 is reachable from this computer, not the network.
    app.run(host="127.0.0.1", port=5000)
