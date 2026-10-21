"""Score the assistant's answers so models (and prompts) can be compared.

Covers the lab's "how to choose the right LLM" criteria with numbers:
- quality: usable JSON, sentiment that matches a human label, rule breaks
- speed: seconds per answer (cost is time when running locally)

The rule checks are simple text patterns. They catch the problems seen in
scripts 04 and 05 (promises, claimed actions, placeholders, asking for
details the customer already gave), but they are not a full review: a
reply can break a rule in words these patterns don't cover.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from support_assistant.schemas import AIResponse

# Same thresholds as the sentiment colors on the web page.
NEGATIVE_BELOW = 35
POSITIVE_ABOVE = 65


def sentiment_bucket(score: int) -> str:
    """Turn a 0-100 score into the label used in the dataset."""
    if score < NEGATIVE_BELOW:
        return "negative"
    if score <= POSITIVE_ABOVE:
        return "neutral"
    return "positive"


# Order numbers (5+ digits, with or without #) and UPS-style tracking numbers.
ID_PATTERN = re.compile(r"\b1Z[0-9A-Z]{16}\b|\d{5,}")


def find_ids(text: str) -> set[str]:
    """Order and tracking numbers in a text, without the # sign."""
    return set(ID_PATTERN.findall(text))


# One pattern per rule in the system prompt. [^.!?]* means "in the same sentence".
RULES = {
    "promise": re.compile(
        r"\b(will|we'll|i'll|shall)\b[^.!?]*\b(refund|reimburs|replac|compensat|credit)"
        r"|\b(refund|replacement|compensation)\b[^.!?]*\bwill be\b",
        re.IGNORECASE,
    ),
    "claimed action": re.compile(
        r"\b(i|we)\s*(have|'ve)\s+(already\s+)?(reported|contacted|processed|issued|refunded|"
        r"escalated|forwarded|passed|sent|checked|cancell?ed|reset|fixed|created|opened|"
        r"submitted|notified|informed)\b",
        re.IGNORECASE,
    ),
    "placeholder": re.compile(r"\[[^\]]{1,40}\]|<[A-Za-z ]{1,30}>|\{[A-Za-z_ ]{1,30}\}"),
}

ASKS_FOR_DETAILS = re.compile(
    r"\b(provide|share|send|confirm|give)\b[^.!?]*\b(order|tracking)\b", re.IGNORECASE
)

# Statements about what the product or company does. Only checked for messages
# that ask a product question the assistant has no information about.
STATES_PRODUCT_FACT = re.compile(
    r"\b(you can|currently|is available|are available|is not available|isn't available|"
    r"we (do|don't|do not) (offer|deliver|support)|(does|do) not support|allows?|supports?)\b",
    re.IGNORECASE,
)


def find_rule_breaks(message: str, answer: AIResponse, product_question: bool) -> list[str]:
    """Names of the rules the customer reply breaks (empty list when none)."""
    breaks = [name for name, pattern in RULES.items() if pattern.search(answer.response)]
    if find_ids(message) and ASKS_FOR_DETAILS.search(answer.response):
        breaks.append("asks for given details")
    if product_question and STATES_PRODUCT_FACT.search(answer.response):
        breaks.append("states product fact")
    return breaks


@dataclass
class ScoreRow:
    """The result of one model on one message."""

    model: str
    message_id: str
    expected_sentiment: str
    attempts: int                  # 1 = first try, 2 = after a retry, 0 = no usable answer
    seconds: float
    sentiment: int | None = None
    rule_breaks: list[str] = field(default_factory=list)
    ids_in_next_action: bool | None = None  # None when the message has no IDs

    @property
    def usable(self) -> bool:
        return self.attempts > 0

    @property
    def sentiment_matches(self) -> bool:
        return self.usable and sentiment_bucket(self.sentiment) == self.expected_sentiment


def score_answer(
    model: str,
    message_id: str,
    message: str,
    expected_sentiment: str,
    product_question: bool,
    answer: AIResponse,
    attempts: int,
    seconds: float,
) -> ScoreRow:
    """Score one usable answer."""
    ids = find_ids(message)
    return ScoreRow(
        model=model,
        message_id=message_id,
        expected_sentiment=expected_sentiment,
        attempts=attempts,
        seconds=seconds,
        sentiment=answer.sentiment,
        rule_breaks=find_rule_breaks(message, answer, product_question),
        ids_in_next_action=ids <= find_ids(answer.next_action) if ids else None,
    )


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def summarize(rows: list[ScoreRow]) -> dict[str, dict]:
    """One line of numbers per model, in the order the models first appear."""
    summary = {}
    for model in dict.fromkeys(row.model for row in rows):
        mine = [row for row in rows if row.model == model]
        usable = [row for row in mine if row.usable]
        with_ids = [row for row in usable if row.ids_in_next_action is not None]
        breaks = Counter(name for row in usable for name in row.rule_breaks)
        summary[model] = {
            "messages": len(mine),
            "first_try": sum(row.attempts == 1 for row in mine),
            "after_retry": sum(row.attempts == 2 for row in mine),
            "failed": sum(row.attempts == 0 for row in mine),
            "sentiment_matches": sum(row.sentiment_matches for row in usable),
            "usable": len(usable),
            "rule_breaks": breaks,
            "replies_with_breaks": sum(bool(row.rule_breaks) for row in usable),
            "ids_kept": sum(bool(row.ids_in_next_action) for row in with_ids),
            "with_ids": len(with_ids),
            "avg_seconds": sum(row.seconds for row in mine) / len(mine),
        }
    return summary


def to_markdown(summary: dict[str, dict], label: str) -> str:
    """The summary as Markdown tables, ready for the README."""
    lines = [
        f"### Scorecard: {label}",
        "",
        "| Model | Usable JSON (1st try + retry) | Sentiment matches label "
        "| Replies with a rule break | IDs kept in next action | Avg seconds |",
        "|---|---|---|---|---|---|",
    ]
    for model, s in summary.items():
        usable = s["first_try"] + s["after_retry"]
        lines.append(
            f"| {model} "
            f"| {percent(usable, s['messages'])} ({s['first_try']} + {s['after_retry']}) "
            f"| {percent(s['sentiment_matches'], s['usable'])} "
            f"| {s['replies_with_breaks']} of {s['usable']} "
            f"| {percent(s['ids_kept'], s['with_ids'])} "
            f"| {s['avg_seconds']:.1f} |"
        )

    names = sorted({name for s in summary.values() for name in s["rule_breaks"]})
    if names:
        lines += ["", "| Rule break | " + " | ".join(summary) + " |"]
        lines.append("|---|" + "---|" * len(summary))
        for name in names:
            counts = " | ".join(str(s["rule_breaks"][name]) for s in summary.values())
            lines.append(f"| {name} | {counts} |")
    return "\n".join(lines) + "\n"
