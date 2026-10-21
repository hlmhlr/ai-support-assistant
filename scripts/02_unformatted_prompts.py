"""Send the same unformatted prompts to all three models.

Aim: the lab's "Trying other LLMs" step, where switching from Granite to Llama
gave a rambling answer. Here all three models get the same plain prompts.
What you'll see: each model's answers, then a table with the numbers the lab
uses to choose a model: quality (the answer), speed and size of the output.
What to notice: some models don't stop after the answer, or continue the
prompt as if it were a document. The prompt has no special tokens, so the
model can't tell where the question ends. The next stage fixes this.
"""

from support_assistant.config import MODELS
from support_assistant.llm import generate_text

params = {"decoding_method": "greedy", "max_new_tokens": 100}

# The lab's prompt, plus a support prompt of the kind this assistant handles.
prompts = {
    "capital": """
Only reply with the answer. What is the capital of Canada?
""",
    "routing": """
Only reply with the answer. A customer writes: "I was charged twice for my order."
Which team should handle this: Billing, Shipping or Technical Support?
""",
}

results = []
for name in MODELS:
    for prompt_id, prompt in prompts.items():
        # Only one model fits in 6 GB of GPU memory, so Ollama swaps models.
        # The first call to each model includes loading time, shown separately.
        result = generate_text(name, prompt, params)
        results.append((prompt_id, result))
        print(f"--- {name} / {prompt_id} ---")
        print(result.text, "\n")

print(f"{'model':<8} {'prompt':<8} {'tokens':>6} {'answer s':>9} {'tok/s':>6} {'load s':>7}")
for prompt_id, r in results:
    print(
        f"{r.model:<8} {prompt_id:<8} {r.output_tokens:>6} "
        f"{r.generation_seconds:>9.1f} {r.tokens_per_second:>6.1f} {r.load_seconds:>7.1f}"
    )

print(
    "\nA short, exact answer uses few tokens. On a paid API every extra token costs "
    "money; locally it costs time."
)
