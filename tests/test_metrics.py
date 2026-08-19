"""Scoring rules, pinned by the cases that decide whether a fine-tune 'won'."""
from __future__ import annotations

from slipft.metrics import Report, field_correct, is_clean_json, parse_json, schema_valid

TRUTH = {
    "doc_type": "transfer",
    "amount": 1234.5,
    "fee": None,
    "date": "2026-08-18",
    "time": "14:32",
    "sender_name": "สมชาย ใจดี",
    "sender_account": "xxx-x-x1234-x",
    "receiver_name": "สุภัทรา สังสิลา",
    "receiver_account": "x7810",
    "channel": "ธนาคารกสิกรไทย",
    "reference": "256584563536782",
}


def test_a_markdown_fence_is_recovered_but_not_counted_as_clean():
    raw = '```json\n{"amount": 12}\n```'
    assert parse_json(raw) == {"amount": 12}
    assert not is_clean_json(raw)
    assert is_clean_json('{"amount": 12}')


def test_prose_around_the_object_is_recovered():
    assert parse_json('Here you go: {"amount": 12} — hope that helps') == {"amount": 12}


def test_a_reply_with_no_object_scores_nothing():
    assert parse_json("I cannot read this slip.") is None


def test_amounts_compare_as_numbers_not_as_strings():
    assert field_correct("amount", 1234.5, "1,234.50")
    assert field_correct("amount", 1234.5, 1234.5)
    assert not field_correct("amount", 1234.5, 1234.0)


def test_a_missing_fee_and_a_zero_fee_are_different_answers():
    assert field_correct("fee", None, None)
    assert not field_correct("fee", None, 0)
    assert not field_correct("fee", 0.0, None)


def test_time_is_compared_after_normalisation():
    assert field_correct("time", "14:32", "14.32")
    assert field_correct("time", "14:32", "14:32 น.")
    assert not field_correct("time", "14:32", "02:32")


def test_account_masking_must_survive_verbatim():
    assert field_correct("sender_account", "xxx-x-x1234-x", " xxx-x-x1234-x ")
    assert not field_correct("sender_account", "xxx-x-x1234-x", "1234")


def test_schema_validation_rejects_a_wrong_doc_type():
    assert schema_valid(dict(TRUTH))
    assert not schema_valid({**TRUTH, "doc_type": "payment"})
    assert not schema_valid({**TRUTH, "amount": "หนึ่งพัน"})


def test_a_perfect_answer_scores_one_everywhere():
    r = Report()
    r.add(TRUTH, '{"doc_type":"transfer","amount":1234.5,"fee":null,"date":"2026-08-18",'
                 '"time":"14:32","sender_name":"สมชาย ใจดี","sender_account":"xxx-x-x1234-x",'
                 '"receiver_name":"สุภัทรา สังสิลา","receiver_account":"x7810",'
                 '"channel":"ธนาคารกสิกรไทย","reference":"256584563536782"}')
    s = r.summary()
    assert s["field_accuracy"] == 1.0
    assert s["exact_record_rate"] == 1.0
    assert s["clean_json_rate"] == 1.0


def test_an_unparseable_answer_costs_every_field():
    r = Report()
    r.add(TRUTH, "ไม่สามารถอ่านสลิปนี้ได้")
    s = r.summary()
    assert s["field_accuracy"] == 0.0
    assert s["parsed_rate"] == 0.0
    assert s["exact_record_rate"] == 0.0


def test_one_wrong_field_costs_the_exact_record_but_not_the_rest():
    r = Report()
    wrong_year = {**TRUTH, "date": "2569-08-18"}
    r.add(TRUTH, __import__("json").dumps(wrong_year, ensure_ascii=False))
    s = r.summary()
    assert s["exact_record_rate"] == 0.0
    assert s["field_accuracy"] == 10 / 11
    assert s["per_field_accuracy"]["date"] == 0.0
    assert s["per_field_accuracy"]["amount"] == 1.0


def test_latency_percentiles_come_from_the_values_given():
    r = Report()
    for ms in (100, 200, 300, 400, 500):
        r.add(TRUTH, "{}", latency_ms=ms, prompt_tokens=10, completion_tokens=5)
    s = r.summary()
    assert s["latency_ms_median"] == 300
    assert s["prompt_tokens_mean"] == 10


def test_channel_accepts_the_names_an_institution_actually_goes_by():
    from slipft.metrics import _norm_channel

    assert field_correct("channel", "ธนาคารไทยพาณิชย์", "SCB")
    assert field_correct("channel", "ธนาคารทหารไทยธนชาต", "ttb touch")
    assert field_correct("channel", "TrueMoney Wallet", "TrueMoney")
    assert field_correct("channel", "Grab", "GrabPay")
    assert field_correct("channel", "ธนาคารกรุงไทย", "ธนาคารกรุงไทย Mobile Banking")
    assert _norm_channel("KBANK") == "ธนาคารกสิกรไทย"


def test_channel_still_separates_two_different_institutions():
    assert not field_correct("channel", "ธนาคารกสิกรไทย", "ธนาคารกรุงไทย")
    assert not field_correct("channel", "Grab", "TrueMoney Wallet")
    assert not field_correct("channel", "เซเว่น อีเลฟเว่น", "แฟมิลี่มาร์ท")
    assert not field_correct("channel", "Café Amazon", None)
