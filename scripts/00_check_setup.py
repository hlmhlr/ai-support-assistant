"""Check that Ollama is running and the three models are downloaded.

Aim: catch setup problems before running anything else.
What you'll see: one line per model (OK or the command to download it),
then a one-word answer from each downloaded model.
"""

import sys

import ollama

from support_assistant.config import MODELS, OLLAMA_BASE_URL
from support_assistant.llm import get_chat_model

client = ollama.Client(host=OLLAMA_BASE_URL)

# 1. Is Ollama running?
try:
    installed = {m.model for m in client.list().models}
except Exception:
    print(f"Can't reach Ollama at {OLLAMA_BASE_URL}. Start the Ollama app and try again.")
    sys.exit(1)
print(f"Ollama is running at {OLLAMA_BASE_URL}\n")

# 2. Are the models downloaded? Ollama adds ":latest" when no tag is given.
missing = []
for name, tag in MODELS.items():
    full_tag = tag if ":" in tag else f"{tag}:latest"
    if full_tag in installed:
        print(f"  OK       {name:<8} {tag}")
    else:
        print(f"  MISSING  {name:<8} run: ollama pull {tag}")
        missing.append(name)

# 3. Does each downloaded model answer?
print()
for name in MODELS:
    if name in missing:
        continue
    llm = get_chat_model(name, {"max_new_tokens": 10})
    answer = llm.invoke("Reply with one word: ready").content.strip()
    print(f"  {name:<8} says: {answer}")

if missing:
    print("\nDownload the missing models, then run this script again.")
    sys.exit(1)
print("\nAll set.")
