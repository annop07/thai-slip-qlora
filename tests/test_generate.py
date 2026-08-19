"""The corpus is only useful if the label is always right, so that is what these pin."""
from __future__ import annotations

import random
import re

import pytest

from slipft.generate import (
    HELD_OUT_TEMPLATES,
    TEMPLATES,
    TRAIN_TEMPLATES,
    build_split,
    sample,
)
from slipft.schema import Slip


def test_same_seed_gives_the_same_corpus():
    a = build_split(40, random.Random(7), TRAIN_TEMPLATES)
    b = build_split(40, random.Random(7), TRAIN_TEMPLATES)
    assert [x.text for x in a] == [x.text for x in b]
    assert [x.record for x in a] == [x.record for x in b]


def test_every_record_validates_against_the_schema():
    for s in build_split(120, random.Random(3), list(TEMPLATES)):
        Slip.model_validate(s.record)


@pytest.mark.parametrize("template", list(TEMPLATES))
def test_the_amount_appears_in_the_text_it_was_rendered_from(template):
    rng = random.Random(11)
    for _ in range(25):
        s = sample(rng, template, noise=False)
        printed = f"{s.record['amount']:,.2f}"
        plain = f"{s.record['amount']:.2f}"
        assert printed in s.text or plain in s.text


@pytest.mark.parametrize("template", list(TEMPLATES))
def test_the_label_date_is_gregorian_and_the_slip_usually_is_not(template):
    """543 years of difference is the single hardest thing in this task.

    The label must be the Gregorian year while the text mostly shows the
    Buddhist one — if the generator ever printed the label verbatim, a model
    could score full marks by copying, and the benchmark would measure nothing.
    """
    rng = random.Random(5)
    buddhist_seen = 0
    for _ in range(40):
        s = sample(rng, template, noise=False)
        year = int(s.record["date"][:4])
        assert 2020 <= year <= 2030
        if str(year + 543) in s.text or f"{(year + 543) % 100:02d}" in s.text:
            buddhist_seen += 1
    assert buddhist_seen > 0


def test_held_out_layouts_are_absent_from_the_training_templates():
    assert set(HELD_OUT_TEMPLATES) == {"ttb_transfer", "grab_receipt"}
    assert not set(TRAIN_TEMPLATES) & set(HELD_OUT_TEMPLATES)
    for s in build_split(60, random.Random(2), TRAIN_TEMPLATES):
        assert not s.held_out


def test_noise_never_touches_a_value():
    """Whitespace damage is allowed; a corrupted digit would make the label wrong."""
    rng = random.Random(19)
    for template in TEMPLATES:
        for _ in range(15):
            clean = sample(random.Random(99), template, noise=False)
            noisy = sample(random.Random(99), template, noise=True)
            assert clean.record == noisy.record
            digits_clean = re.sub(r"\s+", "", re.sub(r"[^\d.,]", "", clean.text))
            digits_noisy = re.sub(r"\s+", "", re.sub(r"[^\d.,]", "", noisy.text))
            assert digits_clean == digits_noisy
            _ = rng.random()


def test_absent_fields_are_null_rather_than_empty():
    for s in build_split(80, random.Random(23), list(TEMPLATES)):
        for value in s.record.values():
            assert value != "" and value != []
        if s.record["doc_type"] == "receipt":
            assert s.record["sender_account"] is None
