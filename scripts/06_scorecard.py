"""Score every model on a set of labelled support messages.

Aim: the lab's "how to choose the right LLM" step, measured instead of judged
by eye: quality (usable JSON, sentiment, rule breaks) and speed.
What you'll see: progress per model, then a Markdown table, also saved to
outputs/scorecard_<label>.md, with the details per message in a CSV file.
What to notice: which model gives usable answers most often, which one
breaks the reply rules, and what that costs in seconds per message.

Usage:
  uv run python scripts/06_scorecard.py --label before
  uv run python scripts/06_scorecard.py --label quick --models granite --limit 3
"""

import argparse
import csv
import time
from pathlib import Path

from support_assistant.config import MODELS
from support_assistant.evaluation import ScoreRow, score_answer, summarize, to_markdown
from support_assistant.llm import generate_text
from support_assistant.model import StructuredOutputError, get_structured_result

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--label", default="run", help="name for this run, used in file names")
parser.add_argument("--models", default=",".join(MODELS), help="comma-separated model names")
parser.add_argument("--limit", type=int, help="only the first N messages (for a quick check)")
parser.add_argument("--data", type=Path, default=ROOT / "data" / "support_messages.csv")
args = parser.parse_args()

models = [name.strip() for name in args.models.split(",")]
with args.data.open(encoding="utf-8") as file:
    messages = list(csv.DictReader(file))[: args.limit]

rows: list[ScoreRow] = []
for model in models:
    # Load the model before timing, so the first message doesn't include loading time.
    generate_text(model, "Hi", {"max_new_tokens": 1})

    for item in messages:
        start = time.perf_counter()
        try:
            result = get_structured_result(model, item["message"])
        except StructuredOutputError:
            row = ScoreRow(
                model=model,
                message_id=item["id"],
                expected_sentiment=item["expected_sentiment"],
                attempts=0,
                seconds=time.perf_counter() - start,
            )
        else:
            row = score_answer(
                model=model,
                message_id=item["id"],
                message=item["message"],
                expected_sentiment=item["expected_sentiment"],
                product_question=item["product_question"] == "yes",
                answer=result.response,
                attempts=result.attempts,
                seconds=time.perf_counter() - start,
            )
        rows.append(row)
        status = {0: "no usable answer", 1: "ok", 2: "ok after retry"}[row.attempts]
        breaks = f" | breaks: {', '.join(row.rule_breaks)}" if row.rule_breaks else ""
        print(f"{model:<8} {item['id']:<17} {status:<16} {row.seconds:5.1f} s{breaks}")

# Save the details and the summary.
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

with (output_dir / f"scorecard_{args.label}.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["model", "message_id", "attempts", "seconds", "sentiment", "expected",
         "sentiment_matches", "rule_breaks", "ids_in_next_action"]
    )
    for row in rows:
        writer.writerow(
            [row.model, row.message_id, row.attempts, f"{row.seconds:.2f}", row.sentiment,
             row.expected_sentiment, row.sentiment_matches, "; ".join(row.rule_breaks),
             row.ids_in_next_action]
        )

markdown = to_markdown(summarize(rows), args.label)
(output_dir / f"scorecard_{args.label}.md").write_text(markdown, encoding="utf-8")
print("\n" + markdown)
print(f"Saved outputs/scorecard_{args.label}.md and outputs/scorecard_{args.label}.csv")
