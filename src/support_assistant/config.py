"""Settings in one place: which models to use and how they generate text.

Replaces the lab's `config.py`. There are no watsonx credentials or project ID:
the models run on your own computer through Ollama.
"""

import os

from dotenv import load_dotenv

# Read settings from the .env file (if it exists) into environment variables.
load_dotenv()

# Where Ollama is running. The second argument is the fallback value.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# The three model families from the lab, in versions that fit a 6 GB GPU.
# The key ("granite", "llama", "mistral") is the name used everywhere else
# in the code; the value is the Ollama model tag.
MODELS = {
    "granite": os.getenv("GRANITE_MODEL", "granite4:tiny-h"),
    "llama": os.getenv("LLAMA_MODEL", "llama3.2:3b"),
    "mistral": os.getenv("MISTRAL_MODEL", "mistral:7b"),
}

# Same settings as the lab's PARAMETERS: greedy decoding, up to 256 new tokens.
PARAMETERS = {
    "decoding_method": "greedy",  # "greedy" = always the most likely token; "sample" = random
    "max_new_tokens": 256,        # maximum length of the answer, in tokens
    "temperature": 0.7,           # randomness, only used when decoding_method is "sample"
}
