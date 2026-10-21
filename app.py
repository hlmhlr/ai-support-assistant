"""Web page and JSON API for the support assistant.

Replaces the lab's `app.py`. GET / serves the page. POST /generate takes a
customer message and a model name and returns the summary, sentiment, reply
and next action. POST /compare sends one message to every model, for the
lab's "choosing the right model" step.

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
from flask import Flask, jsonify, render_template, request

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

    @app.get("/")
    def index():
        # The page lists the models from config, so a new model appears automatically.
        return render_template("index.html", models=MODELS)

    def read_request(need_model: bool):
        """Check the JSON body. Returns (message, model, None) or (None, None, error)."""
        # silent=True returns None instead of raising when the body isn't JSON.
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return None, None, error("Send a JSON body with 'message' and 'model'.", 400)

        message = data.get("message")
        model = data.get("model")
        if not isinstance(message, str) or not message.strip() or (need_model and not model):
            return None, None, error("Missing message or model selection.", 400)
        if need_model and model not in MODELS:
            choices = ", ".join(MODELS)
            return None, None, error(f"Unknown model {model!r}. Choose one of: {choices}.", 400)
        if len(message) > MAX_MESSAGE_LENGTH:
            return None, None, error(f"Message is longer than {MAX_MESSAGE_LENGTH} characters.", 400)
        return message.strip(), model, None

    def run_model(model: str, message: str) -> tuple[dict, int]:
        """Call one model and turn every outcome into (JSON body, HTTP status)."""
        start = time.perf_counter()
        try:
            result = responder(model, message)
        except StructuredOutputError as exc:
            # The model answered, but not in a usable form, even after a retry.
            return {"error": str(exc), "problem": exc.problem}, 422
        except (httpx.TransportError, ConnectionError):
            # LangChain raises httpx errors, the plain ollama client ConnectionError.
            return {"error": "Can't reach Ollama. Start the Ollama app and try again."}, 503
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                return {"error": f"Model not downloaded. Run: ollama pull {MODELS[model]}"}, 503
            app.logger.exception("Ollama returned an error")
            return {"error": "Ollama returned an error. See the server log."}, 503
        except Exception:
            app.logger.exception("Unexpected error while calling %s", model)
            return {"error": "Something went wrong. See the server log."}, 500

        body = {
            **result.model_dump(),
            "model": model,
            "duration": round(time.perf_counter() - start, 2),  # seconds, as in the lab
        }
        return body, 200

    @app.post("/generate")
    def generate():
        message, model, problem = read_request(need_model=True)
        if problem:
            return problem
        body, status = run_model(model, message)
        return jsonify(body), status

    @app.post("/compare")
    def compare():
        """Send one message to every model, one after another, and return all answers.

        One model failing (e.g. an unusable answer) doesn't hide the others:
        each result carries its own status. Only when every model fails on the
        server side (Ollama down) does the whole request fail.
        """
        message, _, problem = read_request(need_model=False)
        if problem:
            return problem

        results = []
        for model in MODELS:
            body, status = run_model(model, message)
            results.append({"model": model, "status": status, **body})

        if all(result["status"] >= 500 for result in results):
            first = results[0]
            return jsonify({"error": first["error"]}), first["status"]
        return jsonify({"results": results})

    return app


app = create_app()

if __name__ == "__main__":
    # Local use only: 127.0.0.1 is reachable from this computer, not the network.
    app.run(host="127.0.0.1", port=5000)
