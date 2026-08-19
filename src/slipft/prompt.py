"""The prompt every configuration is measured through.

The base model and the fine-tuned model see the *same* system prompt. That is a
deliberate handicap on the fine-tuned model — the usual trick is to train on a
short prompt and compare against a long one, which measures prompt length as
much as training. Here the only difference between the two rows of the
benchmark is the weights.
"""
from __future__ import annotations

import json

SYSTEM = """You extract structured data from Thai payment slips and receipts.

Return ONLY a JSON object with exactly these keys:
doc_type, amount, fee, date, time, sender_name, sender_account,
receiver_name, receiver_account, channel, reference

Rules:
- doc_type: "transfer" for bank/wallet transfers, "topup" for wallet top-ups,
  "receipt" for merchant purchase receipts.
- amount is the final total paid; fee is a separately printed transfer fee.
  Both are numbers (no commas, no currency).
- date is ISO Gregorian YYYY-MM-DD. Thai slips print the Buddhist year, which is
  543 years ahead: 2569 -> 2026. A two-digit year like 69 means 2569 -> 2026.
- time is 24-hour HH:MM.
- Account numbers are copied exactly as printed, keeping every masking character
  (x, X, *, -). Do not strip or reformat them.
- channel is the bank, wallet or shop the document comes from, without product
  or payment-method suffixes: "... Mobile Banking" -> the bank name,
  "GrabPay" -> "Grab", "ttb touch" -> the bank name printed on the slip.
- A merchant receipt names a shop, not two parties: put the shop in channel and
  leave sender_name, sender_account, receiver_name and receiver_account null.
- Use null for anything the document does not state. Never guess, never use 0 or
  an empty string to mean missing.
- No explanation, no markdown fence. JSON only."""


def user_message(slip_text: str) -> str:
    return f"Slip text:\n{slip_text}"


def target_message(record: dict) -> str:
    """The assistant turn the model is trained to produce.

    Compact separators and `ensure_ascii=False`: Thai names as \\uXXXX escapes
    would triple the token count of the answer and teach the model an encoding
    it will never be asked to read back.
    """
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def chat_example(slip_text: str, record: dict) -> dict:
    """One training row in the messages format both TRL and Unsloth accept."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(slip_text)},
            {"role": "assistant", "content": target_message(record)},
        ]
    }
