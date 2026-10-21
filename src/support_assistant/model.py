"""LangChain chains: a prompt template piped into a model, one per model family.

Replaces the lab's `model.py`. The structure is the same (initialize_model,
one PromptTemplate per model, get_ai_response with `template | model`), with
two changes:
- The templates come from templates.py, the formats checked in script 03.
- The model is OllamaLLM (text completion) in raw mode. The lab sent its
  special tokens to ChatWatsonx, a chat model, which wraps the prompt in its
  own template a second time. Raw mode sends the prompt exactly as written.
"""

import json
from dataclasses import dataclass
from langchain_core.exceptions import OutputParserException

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaLLM

from pydantic import ValidationError


from support_assistant.config import OLLAMA_BASE_URL
from support_assistant.llm import model_tag, to_ollama_options
from support_assistant.schemas import AIResponse
from support_assistant.templates import TEMPLATES

# Turns the model's text answer into a Python dict, as in the lab.
json_parser = JsonOutputParser(pydantic_object=AIResponse)

# Instructions for the support task. The rules come from the replies in
# script 04: promises, claimed actions and placeholders.
SUPPORT_SYSTEM_PROMPT = """You are an AI assistant helping a customer support team.
Read the customer's message and fill in every field.
Rules for the reply to the customer:
- Be polite and brief (under 80 words).
- Do not promise refunds, replacements or compensation.
- Do not say that you have already done something.
- Do not use placeholders such as [Name].
Put what the support team must do in next_action, not in the reply."""

# A JSON answer with four fields is longer than a plain reply, and an answer
# cut off by the token limit is not valid JSON. 512 leaves room for both.
STRUCTURED_PARAMS = {"max_new_tokens": 512}


def initialize_model(name: str, params: dict | None = None) -> Runnable:
    """Create a text-completion model for "granite", "llama" or "mistral".

    `.bind(raw=True)` adds raw=True to every call, so Ollama does not add the
    model's chat template on top of ours.
    """
    llm = OllamaLLM(
        model=model_tag(name),
        base_url=OLLAMA_BASE_URL,
        **to_ollama_options(params),
    )
    return llm.bind(raw=True)


def build_template(name: str) -> PromptTemplate:
    """Turn a model's format from templates.py into a LangChain PromptTemplate."""
    model_tag(name)  # fail early with a clear message if the name is unknown
    return PromptTemplate(template=TEMPLATES[name], input_variables=["system", "user"])


def build_chain(name: str, llm: BaseLanguageModel | Runnable | None = None) -> Runnable:
    """Pipe the model's prompt template into the model: template | model.

    `llm` can be replaced by a fake model in tests.
    """
    return build_template(name) | (llm or initialize_model(name))


def get_ai_response(
    name: str,
    system_prompt: str,
    user_prompt: str,
    llm: BaseLanguageModel | Runnable | None = None,
) -> str:
    """Fill in the template, send it to the model and return the answer text.

    One function with a model name replaces the lab's llama_response,
    granite_response and mistral_response.
    """
    chain = build_chain(name, llm)
    answer = chain.invoke({"system": system_prompt, "user": user_prompt})
    return answer.strip()


def build_json_chain(name: str, llm: BaseLanguageModel | Runnable | None = None) -> Runnable:
    """The lab's chain with JSON output: template | model | json_parser."""
    model = llm or initialize_model(name, STRUCTURED_PARAMS)
    return build_template(name) | model | json_parser


class StructuredOutputError(Exception):
    """The model's answer could not be used, even after one retry."""

    def __init__(self, model: str, problem: str, raw_output: str):
        super().__init__(f"{model} did not return a usable answer: {problem}")
        self.model = model
        self.problem = problem
        self.raw_output = raw_output  # kept for logging and debugging


def describe_validation_error(error: ValidationError) -> str:
    """Short, readable list of what is wrong, e.g. 'sentiment: must be <= 100'."""
    problems = []
    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"]) or "answer"
        problems.append(f"{field}: {item['msg']}")
    return "; ".join(problems)



@dataclass
class StructuredResult:
    """A checked answer and how many attempts it took (1, or 2 after a retry)."""

    response: AIResponse
    attempts: int


def get_structured_result(
    name: str,
    user_prompt: str,
    system_prompt: str = SUPPORT_SYSTEM_PROMPT,
    llm: BaseLanguageModel | Runnable | None = None,
) -> StructuredResult:
    """Ask for a JSON answer, check it against AIResponse, and retry once if needed.
    The parser's format instructions (a description of the JSON fields) are
    added to the system turn, as the lab does with {format_prompt}. They are
    passed as a value, so the braces in them are not read as placeholders.
    
    With greedy decoding the same prompt gives the same answer, so the retry
    tells the model what was wrong with its first answer.

    """
    chain = build_json_chain(name, llm)
    system = f"{system_prompt}\n\n{json_parser.get_format_instructions()}"
    user = user_prompt

    for attempt in (1, 2):
        try:
            data = chain.invoke({"system": system, "user": user})
            return StructuredResult(AIResponse.model_validate(data), attempt)
        except OutputParserException as error:
            problem = "the answer was not valid JSON"
            raw_output = error.llm_output or ""
        except ValidationError as error:
            problem = describe_validation_error(error)
            raw_output = json.dumps(data)

        user = (
            f"{user_prompt}\n\n"
            f"Your previous answer could not be used ({problem}):\n{raw_output}\n"
            "Answer again with only the JSON object, with every field filled in."
        )

    raise StructuredOutputError(name, problem, raw_output)


def get_structured_response(
    name: str,
    user_prompt: str,
    system_prompt: str = SUPPORT_SYSTEM_PROMPT,
    llm: BaseLanguageModel | Runnable | None = None,
) -> AIResponse:
    """The checked answer only; used by the web app."""
    return get_structured_result(name, user_prompt, system_prompt, llm).response
