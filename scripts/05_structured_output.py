"""Turn customer messages into structured JSON with every model.

Aim: the lab's "Setting up JSON outputs" section, plus its task of adding a
next step for the support representative (the next_action field).
What you'll see: for each message and model, the four fields of AIResponse,
or the reason the answer could not be used after one retry.
What to notice: the app can now read the summary, sentiment, reply and
next action separately. The reply should not promise anything; actions
belong in next_action.
"""


from support_assistant.config import MODELS
from support_assistant.model import StructuredOutputError, get_structured_response

messages = {
    "double charge": (
        "This is the second time you've charged my card twice for the same order "
        "(#48213). I want my money back today or I'm cancelling my subscription."
    ),
    "missing package": (
        "My package was marked as delivered yesterday, but it is not at my door "
        "and my neighbours have not seen it. Tracking number 1Z999AA10123456784."
    ),
    "happy customer": (
        "The new app update is great, the checkout is much faster now! "
        "Is there a way to save two delivery addresses?"
    ),
}

for message_id, message in messages.items():
    print(f"=== {message_id} ===\n{message}\n")
    for name in MODELS:
        try:
            result = get_structured_response(name, message)
        except StructuredOutputError as error:
            print(f"--- {name}: no usable answer ---\n{error.problem}\n")
            continue
        print(f"--- {name} ---")
        for field, value in result.model_dump().items():
            print(f"{field:>11}: {value}")
        print()
