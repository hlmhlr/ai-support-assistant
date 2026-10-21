"""First call to a model: plain text in, plain text out.

Aim: the lab's `capital.py`, running on local Granite instead of watsonx.
What you'll see: Granite's answer to the lab's question, then to a support
question that this assistant will deal with later.
What to notice: greedy decoding returns the same answer every run, and
max_new_tokens=100 keeps answers short (and, on a paid API, cheap).
"""

from support_assistant.llm import generate_text

# Same settings as the lab: always pick the most likely token, at most 100 tokens.
params = {"decoding_method": "greedy", "max_new_tokens": 100}

# The lab's prompt, unchanged, including the newlines around it.
lab_prompt = """
Only reply with the answer. What is the capital of Canada?
"""

# The same kind of instruction, applied to a support task.
support_prompt = """
Only reply with the answer. A customer writes: "I was charged twice for my order."
Which team should handle this: Billing, Shipping or Technical Support?
"""

for prompt in (lab_prompt, support_prompt):
    result = generate_text("granite", prompt, params)
    print("Prompt:", prompt.strip())
    print("Answer:", result.text)
    print(f"({result.output_tokens} tokens in {result.generation_seconds:.1f} s)\n")
