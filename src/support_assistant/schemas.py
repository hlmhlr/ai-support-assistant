"""The shape of the assistant's answer.

Replaces the lab's `AIResponse` class. The lab's three fields are kept, and
`next_action` is added: the step the support representative should take.
Keeping that step out of the customer reply stops the model from promising
actions to the customer, which the replies in script 04 did.


The lab uses this class only to describe the JSON to the model. Here it also
checks the answer: every field present, text not empty, sentiment 0-100.

"""

from pydantic import BaseModel, ConfigDict, Field


class AIResponse(BaseModel):
    # Remove spaces around text values before checking them.
    model_config = ConfigDict(str_strip_whitespace=True)

    summary: str = Field(
        min_length=1,
        description="One-sentence summary of the customer's message",
    )

    sentiment: int = Field(
        ge=0,
        le=100,
        description="Sentiment score from 0 (very negative) to 100 (very positive)",
    )
    response: str = Field(min_length=1, description="Reply to send to the customer")

    next_action: str = Field(
        min_length=1,
        description="Next step the support representative should take to resolve the issue",
    )
