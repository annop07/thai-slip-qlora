"""Scoring.

Two decisions, both of which move the numbers a lot:

**A missing field is a prediction.** `fee: null` on a slip that prints no fee is
correct and is scored as such. Dropping null fields from the denominator would
reward a model that returns three keys and omits the eight it is unsure about.

**The model is graded on the value, not on the spelling of the value.** Amounts
compare numerically (1234.5 == "1,234.50"), dates and times are compared after
the same normalisation, and strings after stripping. Everything else is exact —
an account printed `XXX-XXX8373` must come back with its masking intact, because
that string is what a human matches against their bank app.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from slipft.schema import FIELDS, Slip

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_NUMERIC = re.compile(r"-?[\d,]*\.?\d+")


def parse_json(raw: str) -> dict | None:
    """Recover the JSON object from whatever the model actually emitted.

    Being generous here is the point: a model that returns the right values
    wrapped in a markdown fence has a formatting problem, and lumping that in
    with a wrong amount would hide which one the fine-tune fixed. The two are
    reported separately — `json_valid` counts what needed no recovery at all.
    """
    if not raw:
        return None
    text = raw.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def is_clean_json(raw: str) -> bool:
    """True when the whole reply parses as an object with no recovery."""
    try:
        return isinstance(json.loads(raw.strip()), dict)
    except (json.JSONDecodeError, AttributeError):
        return False


def schema_valid(obj: dict | None) -> bool:
    if obj is None:
        return False
    try:
        Slip.model_validate(obj)
    except Exception:  # noqa: BLE001 — any validation failure is the same verdict here
        return False
    return True


def _as_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = _NUMERIC.search(value.replace(",", ""))
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


# Institutions answer to more than one name, and none of the alternatives is a
# reading error: a model that returns "SCB" for a slip headed ธนาคารไทยพาณิชย์ has
# read the slip. Scoring `channel` by exact string would mostly measure whether a
# model happens to share this corpus's naming convention — and a fine-tuned model
# learns that convention in the first fifty examples, so the gap it produced would
# be an artifact of the labels rather than a difference in extraction.
_CHANNEL_ALIASES: dict[str, set[str]] = {
    "ธนาคารไทยพาณิชย์": {"scb", "scb easy", "ไทยพาณิชย์"},
    "ธนาคารกสิกรไทย": {"kbank", "kbk", "k plus", "กสิกรไทย"},
    "ธนาคารกรุงไทย": {"ktb", "krungthai", "krungthai next", "กรุงไทย"},
    "ธนาคารกรุงเทพ": {"bbl", "bualuang", "กรุงเทพ"},
    "ธนาคารออมสิน": {"gsb", "ออมสิน"},
    "ธนาคารกรุงศรีอยุธยา": {"bay", "krungsri", "กรุงศรี", "กรุงศรีอยุธยา"},
    "ธนาคารทหารไทยธนชาต": {"ttb", "tmbthanachart", "ทหารไทยธนชาต"},
    "TrueMoney Wallet": {"truemoney", "true money", "true wallet", "truemoney wallet"},
    "Grab": {"grab", "grabpay", "grab pay"},
    "พร้อมเพย์": {"promptpay", "prompt pay"},
}

# Product and channel suffixes that name how the transaction was made rather than
# who it was made with.
_CHANNEL_NOISE = (" mobile banking", " touch", " atm", " online", " application", " app")


def _norm_channel(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    for suffix in _CHANNEL_NOISE:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    text = text.replace("ธนาคาร", "").strip()
    text = " ".join(text.split())
    for canonical, aliases in _CHANNEL_ALIASES.items():
        pool = {canonical.lower().replace("ธนาคาร", "").strip()} | aliases
        if text in {a.replace("ธนาคาร", "").strip() for a in pool}:
            return canonical
    return text


def _norm_time(value) -> str | None:
    if not isinstance(value, str):
        return None
    m = re.search(r"(\d{1,2})[:.](\d{2})", value)
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else value.strip()


def field_correct(name: str, truth, pred) -> bool:
    if name in ("amount", "fee"):
        t, p = _as_number(truth), _as_number(pred)
        if t is None or p is None:
            return t is None and p is None
        return abs(t - p) < 0.005
    if name == "time":
        return _norm_time(truth) == _norm_time(pred)
    if name == "channel":
        return _norm_channel(truth) == _norm_channel(pred)
    if truth is None or pred is None:
        return truth is None and pred is None
    return str(truth).strip() == str(pred).strip()


@dataclass
class Report:
    n: int = 0
    clean_json: int = 0
    parsed: int = 0
    valid_schema: int = 0
    exact_records: int = 0
    per_field: dict[str, int] = field(default_factory=lambda: {f: 0 for f in FIELDS})
    latency_ms: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)

    def add(self, truth: dict, raw: str, latency_ms: float | None = None,
            prompt_tokens: int | None = None, completion_tokens: int | None = None) -> None:
        self.n += 1
        if latency_ms is not None:
            self.latency_ms.append(latency_ms)
        if prompt_tokens is not None:
            self.prompt_tokens.append(prompt_tokens)
        if completion_tokens is not None:
            self.completion_tokens.append(completion_tokens)

        if is_clean_json(raw):
            self.clean_json += 1
        pred = parse_json(raw)
        if pred is None:
            # Nothing parsed: every field counts as wrong. No partial credit for
            # prose that happens to contain the right number.
            return
        self.parsed += 1
        if schema_valid(pred):
            self.valid_schema += 1

        hits = 0
        for name in FIELDS:
            if field_correct(name, truth.get(name), pred.get(name)):
                self.per_field[name] += 1
                hits += 1
        if hits == len(FIELDS):
            self.exact_records += 1

    @property
    def field_accuracy(self) -> float:
        return sum(self.per_field.values()) / (self.n * len(FIELDS)) if self.n else 0.0

    @property
    def exact_rate(self) -> float:
        return self.exact_records / self.n if self.n else 0.0

    def _p(self, values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(int(q * len(ordered)), len(ordered) - 1)]

    def summary(self) -> dict:
        return {
            "n": self.n,
            "clean_json_rate": self.clean_json / self.n if self.n else 0.0,
            "parsed_rate": self.parsed / self.n if self.n else 0.0,
            "schema_valid_rate": self.valid_schema / self.n if self.n else 0.0,
            "field_accuracy": self.field_accuracy,
            "exact_record_rate": self.exact_rate,
            "per_field_accuracy": {
                k: (v / self.n if self.n else 0.0) for k, v in self.per_field.items()
            },
            "latency_ms_median": self._p(self.latency_ms, 0.5),
            "latency_ms_p90": self._p(self.latency_ms, 0.9),
            "prompt_tokens_mean": (
                sum(self.prompt_tokens) / len(self.prompt_tokens) if self.prompt_tokens else 0.0
            ),
            "completion_tokens_mean": (
                sum(self.completion_tokens) / len(self.completion_tokens)
                if self.completion_tokens else 0.0
            ),
        }
