"""The extraction target.

One schema, used in four places: to generate the ground truth, to describe the
task to the model in the prompt, to validate whatever the model returns, and to
score it field by field. Keeping them the same object is what makes "the model
answered in the wrong shape" a measurable failure rather than a parsing bug.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["transfer", "topup", "receipt"]

# Scored in this order in the report. Every field is scored, including the ones
# that are frequently absent — a model that invents a fee when the slip has none
# is wrong in the same way as one that misses a fee that is there.
FIELDS: tuple[str, ...] = (
    "doc_type",
    "amount",
    "fee",
    "date",
    "time",
    "sender_name",
    "sender_account",
    "receiver_name",
    "receiver_account",
    "channel",
    "reference",
)


class Slip(BaseModel):
    """What one slip or receipt says, normalised.

    Normalisation is the part a base model gets wrong most often, and it is
    deliberately not something the prompt can hand-hold its way through:

    * `date` is ISO Gregorian. Thai slips print the Buddhist year (2569), often
      with an abbreviated Thai month ("18 ส.ค. 69"), so producing 2026-08-18
      means subtracting 543 *and* reading a month name that is not a number.
    * `amount` and `fee` are numbers, not the "1,234.50 บาท" that is printed.
    * absent means null, never 0 and never "".
    """

    doc_type: DocType
    amount: float = Field(description="Total in THB, digits only")
    fee: float | None = Field(default=None, description="Transfer fee in THB, null when not printed")
    date: str = Field(description="ISO Gregorian date, YYYY-MM-DD")
    time: str | None = Field(default=None, description="24-hour HH:MM, null when not printed")
    sender_name: str | None = None
    sender_account: str | None = Field(default=None, description="As printed, masking included")
    receiver_name: str | None = None
    receiver_account: str | None = None
    channel: str | None = Field(default=None, description="Bank, wallet or merchant the document comes from")
    reference: str | None = Field(default=None, description="Reference / transaction id as printed")
