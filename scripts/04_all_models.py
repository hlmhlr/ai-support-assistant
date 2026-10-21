"""Run one support request through the LangChain chain of every model.

Aim: the lab's sanity check (`llm_test.py` with `call_all_models`), using
the chains from model.py.
What you'll see: each model's reply to the same customer message, with the
time it took.
What to notice: the chain fills the template and calls the model in one step,
and every model answers as a turn of its own, without continuing the prompt.
The replies still differ in length and style; the next stage asks for JSON
so the app can rely on their shape.
"""

import time

from support_assistant.config import MODELS
from support_assistant.model import get_ai_response


def call_all_models(system_prompt: str, user_prompt: str) -> None:
    """Send the same prompts to every model and print the replies."""
    for name in MODELS:
        start = time.perf_counter()
        answer = get_ai_response(name, system_prompt, user_prompt)
        seconds = time.perf_counter() - start
        print(f"--- {name} ({seconds:.1f} s, including loading) ---\n{answer}\n")


call_all_models(
    "You are a friendly customer support assistant. Keep replies under 80 words.",
    'A customer writes: "My package was marked as delivered yesterday, but it is '
    'not at my door and my neighbours have not seen it." Write a reply to the customer.',
)
