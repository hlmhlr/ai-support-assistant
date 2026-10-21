"""Each model's chat format, written out with its special tokens.

Covers the lab's "Special tokens and prompt formatting" section. Every model
family marks roles (system, user, assistant) and the end of a turn with its
own special tokens. A prompt without them looks like a document to continue,
which is why the unformatted prompts in script 02 kept on writing.

These formats are copied from the templates Ollama uses for each model
(`ollama show --template <model>`), for the case of one system message and
one user message. Differences from the lab's templates:
- Llama 3.2 uses <|start_header_id|> and <|eot_id|>. The lab's <|header_start|>
  and <|eot|> belong to Llama 4, a different model.
- Granite 4 uses <|start_of_role|> ... <|end_of_text|>, not <|system|>/<|user|>.
- No <|begin_of_text|> or <s> at the start: Ollama's tokenizer adds that start
  token itself, so writing it by hand would send it twice.
"""

# {system} and {user} are filled in by format_prompt().
TEMPLATES = {
    # IBM Granite 4: each turn is <|start_of_role|>ROLE<|end_of_role|>TEXT<|end_of_text|>
    "granite": (
        "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n"
        "<|start_of_role|>user<|end_of_role|>{user}<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    ),
    # Meta Llama 3.2: role headers, a blank line, the text, then <|eot_id|> (end of turn).
    # Ollama's template always puts the knowledge-date line in the system turn.
    "llama": (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\n\n"
        "{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    # Mistral 7B: no system role. The system text goes inside the instruction,
    # before the user text, separated by a blank line.
    "mistral": "[INST] {system}\n\n{user}[/INST]",
}


def format_prompt(name: str, system: str, user: str) -> str:
    """Wrap a system and a user message in the special tokens of one model."""
    if name not in TEMPLATES:
        raise ValueError(f"No template for {name!r}. Choose one of: {', '.join(TEMPLATES)}")
    return TEMPLATES[name].format(system=system, user=user)
