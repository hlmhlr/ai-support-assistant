"""Unformatted vs formatted prompts, for all three models.

Aim: the lab's "Special tokens and prompt formatting" section, and its task of
trying the special tokens on each model.
What you'll see: for each model and question, three answers:
  1. unformatted  - plain text, as in script 02
  2. formatted    - the same request wrapped in the model's special tokens (templates.py)
  3. chat         - the same request sent as chat messages; Ollama adds the tokens
What to notice:
- Formatted answers stop after the answer: the model knows where its turn ends.
- Formatted and chat answers should match. Same tokens in, and greedy decoding,
  means the same answer out. A match confirms the hand-written templates.
"""

from support_assistant.config import MODELS
from support_assistant.llm import chat_text, generate_text
from support_assistant.templates import format_prompt

params = {"decoding_method": "greedy", "max_new_tokens": 100}

# The instruction moves into the system role, which is what roles are for.
system = "You are a customer support assistant. Reply with the answer only, no explanation."

questions = {
    "capital": "What is the capital of Canada?",
    "routing": (
        'A customer writes: "I was charged twice for my order." '
        "Which team should handle this: Billing, Shipping or Technical Support?"
    ),
}

rows = []
for name in MODELS:
    for question_id, question in questions.items():
        unformatted = generate_text(name, f"{system}\n{question}", params)
        formatted = generate_text(name, format_prompt(name, system, question), params)
        chat = chat_text(name, system, question, params)

        print(f"--- {name} / {question_id} ---")
        for label, result in (("unformatted", unformatted), ("formatted", formatted), ("chat", chat)):
            print(f"[{label}] {result.text}")
        print()

        rows.append((name, question_id, unformatted, formatted, chat))

print(f"{'model':<8} {'question':<8} {'unformatted':>11} {'formatted':>9} {'chat':>5}  same?")
for name, question_id, unformatted, formatted, chat in rows:
    same = "yes" if formatted.text == chat.text else "no"
    print(
        f"{name:<8} {question_id:<8} {unformatted.output_tokens:>11} "
        f"{formatted.output_tokens:>9} {chat.output_tokens:>5}  {same}"
    )
print("\nColumns 3-5 are output tokens. 'same?' compares the formatted and chat answers.")
