"""The shape of the assistant's answer.

Replaces the lab's `AIResponse` class. The lab's three fields are kept, and
`next_action` is added: the step the support representative should take.
Keeping that step out of the customer reply stops the model from promising
actions to the customer, which the replies in script 04 did.
"""

from pydantic import BaseModel, Field


class AIResponse(BaseModel):
    summary: str = Field(description="One-sentence summary of the customer's message")
    sentiment: int = Field(
        description="Sentiment score from 0 (very negative) to 100 (very positive)"
    )
    response: str = Field(description="Reply to send to the customer")
    next_action: str = Field(
        description="Next step the support representative should take to resolve the issue"
    )
