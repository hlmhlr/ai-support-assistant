"""Tests for evaluation.py: sentiment buckets, rule checks and the summary."""

import pytest

from support_assistant.evaluation import (
    ScoreRow,
    find_ids,
    find_rule_breaks,
    score_answer,
    sentiment_bucket,
    summarize,
    to_markdown,
)
from support_assistant.schemas import AIResponse


def answer(response="Sorry about this. Our billing team will review it.", next_action="Check it."):
    return AIResponse(summary="s", sentiment=20, response=response, next_action=next_action)


@pytest.mark.parametrize(
    ("score", "bucket"),
    [(0, "negative"), (34, "negative"), (35, "neutral"), (65, "neutral"), (66, "positive")],
)
def test_sentiment_buckets_match_the_web_page_colors(score, bucket):
    assert sentiment_bucket(score) == bucket


def test_ids_are_found_with_or_without_hash():
    text = "Order #48213, tracking 1Z999AA10123456784, and order 50211."
    assert find_ids(text) == {"48213", "1Z999AA10123456784", "50211"}


@pytest.mark.parametrize(
    ("reply", "rule"),
    [
        ("We will refund the double charge today.", "promise"),
        ("A replacement will be sent tomorrow.", "promise"),
        ("I've reported this to our delivery team.", "claimed action"),
        ("We have already processed your request.", "claimed action"),
        ("Dear [Customer Name], sorry for the delay.", "placeholder"),
    ],
)
def test_each_rule_break_is_detected(reply, rule):
    assert rule in find_rule_breaks("Hi", answer(reply), product_question=False)


def test_a_careful_reply_breaks_no_rule():
    reply = "I'm sorry about the double charge. Our billing team will review it and reply to you."
    assert find_rule_breaks("Order #48213 was charged twice", answer(reply), False) == []


def test_asking_for_an_id_the_customer_gave_is_a_break():
    reply = "Please provide your order number so we can check."
    assert "asks for given details" in find_rule_breaks("Order #48213 is late", answer(reply), False)
    # Without an ID in the message, asking for it is fine.
    assert find_rule_breaks("My order is late", answer(reply), False) == []


def test_product_facts_are_only_checked_for_product_questions():
    reply = "Yes, you can save several addresses in your settings."
    assert "states product fact" in find_rule_breaks("Can I save two?", answer(reply), True)
    assert find_rule_breaks("Can I save two?", answer(reply), False) == []


def test_score_answer_checks_ids_in_next_action():
    kept = score_answer("granite", "x", "Order #48213", "negative", False,
                        answer(next_action="Refund order 48213 if confirmed."), 1, 2.0)
    lost = score_answer("granite", "x", "Order #48213", "negative", False,
                        answer(next_action="Contact the customer."), 1, 2.0)
    no_ids = score_answer("granite", "x", "Thanks!", "positive", False, answer(), 1, 2.0)

    assert kept.ids_in_next_action is True
    assert lost.ids_in_next_action is False
    assert no_ids.ids_in_next_action is None  # nothing to keep


def make_rows():
    return [
        ScoreRow("granite", "a", "negative", 1, 2.0, sentiment=20),
        ScoreRow("granite", "b", "positive", 2, 4.0, sentiment=90, rule_breaks=["promise"]),
        ScoreRow("llama", "a", "negative", 0, 6.0),  # no usable answer
        ScoreRow("llama", "b", "positive", 1, 2.0, sentiment=0,
                 rule_breaks=["promise", "placeholder"], ids_in_next_action=False),
    ]


def test_summary_counts_per_model():
    summary = summarize(make_rows())

    granite, llama = summary["granite"], summary["llama"]
    assert (granite["first_try"], granite["after_retry"], granite["failed"]) == (1, 1, 0)
    assert granite["sentiment_matches"] == 2
    assert granite["avg_seconds"] == 3.0
    assert llama["failed"] == 1
    assert llama["sentiment_matches"] == 0
    assert llama["replies_with_breaks"] == 1  # one reply, even with two broken rules
    assert llama["rule_breaks"]["placeholder"] == 1


def test_markdown_has_a_row_per_model_and_per_rule():
    markdown = to_markdown(summarize(make_rows()), "before")

    assert "### Scorecard: before" in markdown
    assert "| granite | 100% (1 + 1) | 100% | 1 of 2 |" in markdown
    assert "| llama | 50% (1 + 0) | 0% | 1 of 1 |" in markdown
    assert "| placeholder | 0 | 1 |" in markdown
