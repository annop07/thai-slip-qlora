"""Synthetic Thai slip corpus.

The data is generated, and the README says so. Real slips carry account numbers
and names that cannot go in a public repository, and hand-labelling a thousand
of them was not going to happen in a week. What generation buys, besides
volume, is a ground truth that is correct by construction: the value is sampled
first and the slip is rendered around it, so a disagreement between the model
and the label is always the model.

What it costs is realism, and the honest reading of every number in the
benchmark is "on slips that look like these". Two things are done to stop that
from being a free pass:

* **Two layouts never appear in training.** `ttb` transfers and Grab receipts
  are generated for the test split only, so the report can separate "learned
  the task" from "memorised nine layouts".
* **Values are drawn independently of layout.** A date format is not tied to a
  bank, so the model cannot shortcut Buddhist-year conversion by recognising
  which bank printed the slip.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from slipft.prompt import chat_example

THAI_MONTHS_ABBR = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
                    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
THAI_MONTHS_FULL = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
                    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]

FIRST_NAMES = ["สมชาย", "สุภัทรา", "อานนท์", "ธนกฤต", "ปิยะดา", "วรรณา", "ณัฐพงษ์", "กิตติศักดิ์",
               "พรทิพย์", "ชลธิชา", "ศิริพร", "ภาณุพงศ์", "อรทัย", "จิราพร", "เอกชัย", "นภาพร",
               "ธีรภัทร", "ปาริชาต", "สุรศักดิ์", "มนัสวี"]
LAST_NAMES = ["ใจดี", "แสงสว่าง", "ทองคำ", "สังสิลา", "บุญมี", "รักไทย", "ศรีสุข", "วงศ์วิวัฒน์",
              "พูนทรัพย์", "อินทร์แก้ว", "เกษมสุข", "ชูเกียรติ", "มั่นคง", "ธนบดี"]

BANKS = [
    ("ธนาคารไทยพาณิชย์", "SCB"),
    ("ธนาคารกสิกรไทย", "KBANK"),
    ("ธนาคารกรุงไทย", "KTB"),
    ("ธนาคารกรุงเทพ", "BBL"),
    ("ธนาคารออมสิน", "GSB"),
    ("ธนาคารกรุงศรีอยุธยา", "BAY"),
]

SHOP_ITEMS = [
    ("นมสด 1 ลิตร", 59.0), ("ข้าวกล่องอุ่นร้อน", 45.0), ("กาแฟกระป๋อง", 25.0),
    ("บะหมี่กึ่งสำเร็จรูป", 6.0), ("น้ำดื่ม 600 มล.", 7.0), ("ขนมปังไส้สังขยา", 22.0),
    ("ลาเต้เย็น", 65.0), ("อเมริกาโน่ร้อน", 55.0), ("ครัวซองต์", 45.0),
    ("เค้กช็อกโกแลต", 89.0), ("ชาเขียวนม", 60.0),
]


@dataclass(frozen=True)
class Sample:
    text: str
    record: dict
    template: str
    held_out: bool


# --- value sampling -----------------------------------------------------------


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _masked_account(rng: random.Random) -> str:
    """Account as printed. Six masking conventions, none of them normalised away."""
    tail = f"{rng.randint(0, 9999):04d}"
    style = rng.randrange(6)
    if style == 0:
        return f"xxx-x-x{tail}-x"
    if style == 1:
        return f"xxx-x-{tail}-x"
    if style == 2:
        return f"x{tail}"
    if style == 3:
        return f"{rng.randint(100, 999)}-{rng.randint(0, 9)}-{tail}-{rng.randint(0, 9)}"
    if style == 4:
        return f"XXX-XXX{tail}"
    return f"**{tail}"


def _phone_account(rng: random.Random) -> str:
    return f"0{rng.choice(['6', '8', '9'])}x-xxx-{rng.randint(0, 9999):04d}"


def _sample_date(rng: random.Random) -> date:
    # A window wide enough that year rollover and every month appear.
    return date(2024, 1, 1) + timedelta(days=rng.randrange(0, 3 * 365))


def _render_date(d: date, rng: random.Random) -> str:
    """Six printed forms of the same day, five of them in the Buddhist era."""
    be = d.year + 543
    style = rng.randrange(6)
    if style == 0:
        return f"{d.day} {THAI_MONTHS_ABBR[d.month - 1]} {be}"
    if style == 1:
        return f"{d.day} {THAI_MONTHS_FULL[d.month - 1]} {be}"
    if style == 2:
        return f"{d.day:02d}/{d.month:02d}/{be}"
    if style == 3:
        return f"{d.day} {THAI_MONTHS_ABBR[d.month - 1]} {be % 100:02d}"
    if style == 4:
        return f"{d.day:02d}-{d.month:02d}-{be}"
    # The one Gregorian form, so the model cannot learn "always subtract 543".
    return d.strftime("%d/%m/%Y")


def _render_time(hh: int, mm: int, rng: random.Random) -> str:
    style = rng.randrange(4)
    if style == 0:
        return f"{hh:02d}:{mm:02d}"
    if style == 1:
        return f"{hh:02d}:{mm:02d} น."
    if style == 2:
        return f"{hh:02d}:{mm:02d}:{rng.randint(0, 59):02d}"
    return f"{hh:02d}.{mm:02d} น."


def _render_amount(value: float, rng: random.Random) -> str:
    style = rng.randrange(4)
    body = f"{value:,.2f}"
    if style == 0:
        return body
    if style == 1:
        return f"{body} บาท"
    if style == 2:
        return f"฿{body}"
    return f"{value:.2f}"


def _reference(rng: random.Random) -> str:
    style = rng.randrange(3)
    if style == 0:
        return "".join(rng.choice("0123456789") for _ in range(15))
    if style == 1:
        return f"{rng.choice(['SCB', 'KBK', 'TXN', 'REF'])}{rng.randint(10**9, 10**10 - 1)}"
    return f"{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"


def _base_record(rng: random.Random) -> tuple[dict, date, int, int]:
    d = _sample_date(rng)
    hh, mm = rng.randrange(0, 24), rng.randrange(0, 60)
    record = {
        "doc_type": "transfer",
        "amount": 0.0,
        "fee": None,
        "date": d.isoformat(),
        "time": f"{hh:02d}:{mm:02d}",
        "sender_name": None,
        "sender_account": None,
        "receiver_name": None,
        "receiver_account": None,
        "channel": None,
        "reference": None,
    }
    return record, d, hh, mm


# --- layouts ------------------------------------------------------------------
#
# Each returns the printed text and the record it was rendered from. The record
# is the label: no parsing happens anywhere in this file.


def _bank_transfer(rng: random.Random, bank_index: int | None = None) -> tuple[str, dict]:
    record, d, hh, mm = _base_record(rng)
    bank_th, _ = BANKS[bank_index if bank_index is not None else rng.randrange(len(BANKS))]
    amount = round(rng.uniform(20, 50_000), 2)
    has_fee = rng.random() < 0.35
    fee = round(rng.choice([0.0, 5.0, 10.0, 15.0, 25.0]), 2) if has_fee else None

    record.update(
        doc_type="transfer",
        amount=amount,
        fee=fee,
        sender_name=_person(rng),
        sender_account=_masked_account(rng),
        receiver_name=_person(rng),
        receiver_account=_masked_account(rng),
        channel=bank_th,
        reference=_reference(rng),
    )

    lines = [
        bank_th,
        "โอนเงินสำเร็จ",
        f"วันที่ {_render_date(d, rng)}  เวลา {_render_time(hh, mm, rng)}",
        "จาก",
        f"{record['sender_name']}",
        f"{record['sender_account']}",
        "ไปยัง",
        f"{record['receiver_name']}",
        f"{record['receiver_account']}",
        f"จำนวนเงิน {_render_amount(amount, rng)}",
    ]
    if fee is not None:
        lines.append(f"ค่าธรรมเนียม {_render_amount(fee, rng)}")
    lines.append(f"รหัสอ้างอิง: {record['reference']}")
    return "\n".join(lines), record


def _promptpay(rng: random.Random) -> tuple[str, dict]:
    record, d, hh, mm = _base_record(rng)
    amount = round(rng.uniform(20, 5_000), 2)
    record.update(
        doc_type="transfer",
        amount=amount,
        fee=None,
        sender_name=_person(rng),
        sender_account=_masked_account(rng),
        receiver_name=_person(rng),
        receiver_account=_phone_account(rng),
        channel="พร้อมเพย์",
        reference=_reference(rng),
    )
    text = "\n".join([
        "สลิปโอนเงิน พร้อมเพย์",
        f"{_render_date(d, rng)} {_render_time(hh, mm, rng)}",
        f"ผู้โอน  {record['sender_name']}  {record['sender_account']}",
        f"ผู้รับ   {record['receiver_name']}  {record['receiver_account']}",
        f"ยอดเงิน  {_render_amount(amount, rng)}",
        "ค่าธรรมเนียม  ไม่มี",
        f"เลขที่รายการ {record['reference']}",
    ])
    return text, record


def _truemoney_topup(rng: random.Random) -> tuple[str, dict]:
    record, d, hh, mm = _base_record(rng)
    amount = float(rng.choice([50, 100, 200, 300, 500, 1000]))
    record.update(
        doc_type="topup",
        amount=amount,
        fee=None,
        sender_name=None,
        sender_account=None,
        receiver_name=_person(rng),
        receiver_account=_phone_account(rng),
        channel="TrueMoney Wallet",
        reference=_reference(rng),
    )
    text = "\n".join([
        "TrueMoney Wallet",
        "เติมเงินเข้าวอลเล็ทสำเร็จ",
        f"เบอร์ {record['receiver_account']}  ({record['receiver_name']})",
        f"จำนวน {_render_amount(amount, rng)}",
        f"{_render_date(d, rng)}  {_render_time(hh, mm, rng)}",
        f"Ref. {record['reference']}",
    ])
    return text, record


def _wallet_transfer(rng: random.Random) -> tuple[str, dict]:
    record, d, hh, mm = _base_record(rng)
    amount = round(rng.uniform(20, 3_000), 2)
    record.update(
        doc_type="transfer",
        amount=amount,
        fee=None,
        sender_name=_person(rng),
        sender_account=_phone_account(rng),
        receiver_name=_person(rng),
        receiver_account=_phone_account(rng),
        channel="TrueMoney Wallet",
        reference=_reference(rng),
    )
    text = "\n".join([
        "โอนเงินระหว่างวอลเล็ท - TrueMoney Wallet",
        f"ผู้ส่ง {record['sender_name']} {record['sender_account']}",
        f"ผู้รับ {record['receiver_name']} {record['receiver_account']}",
        f"ยอด {_render_amount(amount, rng)}",
        f"วันเวลา {_render_date(d, rng)} {_render_time(hh, mm, rng)}",
        f"หมายเลขอ้างอิง {record['reference']}",
    ])
    return text, record


def _convenience_receipt(rng: random.Random) -> tuple[str, dict]:
    record, d, hh, mm = _base_record(rng)
    shop = rng.choice(["เซเว่น อีเลฟเว่น", "โลตัส โก แฟรช", "แฟมิลี่มาร์ท", "ซีเจ มอร์"])
    picked = [rng.choice(SHOP_ITEMS) for _ in range(rng.randint(2, 5))]
    total = round(sum(price for _, price in picked), 2)
    record.update(
        doc_type="receipt",
        amount=total,
        fee=None,
        sender_name=None,
        sender_account=None,
        receiver_name=None,
        receiver_account=None,
        channel=shop,
        reference=_reference(rng) if rng.random() < 0.6 else None,
    )
    lines = [shop, f"สาขา {rng.choice(['ขอนแก่น', 'มหาวิทยาลัยขอนแก่น', 'ศรีจันทร์', 'บ้านไผ่'])}",
             f"{_render_date(d, rng)} {_render_time(hh, mm, rng)}", "-" * 24]
    lines += [f"{name:<22}{price:>7,.2f}" for name, price in picked]
    lines += ["-" * 24, f"รวมทั้งสิ้น {_render_amount(total, rng)}", "เงินสด"]
    if record["reference"] is not None:
        lines.append(f"เลขที่ใบเสร็จ {record['reference']}")
    return "\n".join(lines), record


def _cafe_receipt(rng: random.Random) -> tuple[str, dict]:
    record, d, hh, mm = _base_record(rng)
    shop = rng.choice(["Café Amazon", "Inthanin Coffee", "ร้านกาแฟบ้านสวน", "Punthai Coffee"])
    picked = [rng.choice(SHOP_ITEMS[-5:]) for _ in range(rng.randint(1, 3))]
    total = round(sum(price for _, price in picked), 2)
    record.update(
        doc_type="receipt",
        amount=total,
        fee=None,
        receiver_name=None,
        channel=shop,
        reference=_reference(rng) if rng.random() < 0.5 else None,
    )
    lines = [f"** {shop} **", f"ใบเสร็จรับเงิน  {_render_date(d, rng)}  {_render_time(hh, mm, rng)}"]
    lines += [f"{name} x1   {price:,.2f}" for name, price in picked]
    lines.append(f"ยอดสุทธิ {_render_amount(total, rng)}")
    lines.append("ชำระโดย QR พร้อมเพย์")
    if record["reference"] is not None:
        lines.append(f"REF {record['reference']}")
    return "\n".join(lines), record


def _atm_transfer(rng: random.Random) -> tuple[str, dict]:
    """An ATM slip: uppercase-ish, no sender name, fee almost always present."""
    record, d, hh, mm = _base_record(rng)
    bank_th, bank_en = BANKS[rng.randrange(len(BANKS))]
    amount = float(rng.randrange(100, 40_000, 100))
    fee = float(rng.choice([0, 10, 20]))
    record.update(
        doc_type="transfer",
        amount=amount,
        fee=fee,
        sender_name=None,
        sender_account=_masked_account(rng),
        receiver_name=_person(rng),
        receiver_account=_masked_account(rng),
        channel=bank_th,
        reference=_reference(rng),
    )
    text = "\n".join([
        f"{bank_en} ATM",
        f"{bank_th}",
        f"DATE {_render_date(d, rng)}  TIME {_render_time(hh, mm, rng)}",
        f"FROM A/C {record['sender_account']}",
        f"TO {record['receiver_name']}",
        f"A/C {record['receiver_account']}",
        f"AMOUNT {_render_amount(amount, rng)}",
        f"FEE {_render_amount(fee, rng)}",
        f"TRACE {record['reference']}",
    ])
    return text, record


def _mobile_banking_dense(rng: random.Random) -> tuple[str, dict]:
    """A label:value block with the fields in an unhelpful order."""
    record, d, hh, mm = _base_record(rng)
    bank_th, _ = BANKS[rng.randrange(len(BANKS))]
    amount = round(rng.uniform(50, 20_000), 2)
    fee = 0.0 if rng.random() < 0.5 else None
    record.update(
        doc_type="transfer",
        amount=amount,
        fee=fee,
        sender_name=_person(rng),
        sender_account=_masked_account(rng),
        receiver_name=_person(rng),
        receiver_account=_masked_account(rng),
        channel=bank_th,
        reference=_reference(rng),
    )
    rows = [
        f"อ้างอิง: {record['reference']}",
        f"ผู้รับเงิน: {record['receiver_name']} ({record['receiver_account']})",
        f"จำนวนเงิน: {_render_amount(amount, rng)}",
        f"ผู้จ่ายเงิน: {record['sender_name']} ({record['sender_account']})",
        f"วันที่ทำรายการ: {_render_date(d, rng)} {_render_time(hh, mm, rng)}",
        f"ช่องทาง: {bank_th} Mobile Banking",
    ]
    if fee is not None:
        rows.insert(2, f"ค่าธรรมเนียม: {_render_amount(fee, rng)}")
    rng.shuffle(rows)
    return "รายละเอียดรายการ\n" + "\n".join(rows), record


def _ttb_transfer(rng: random.Random) -> tuple[str, dict]:
    """HELD OUT of training. Same task, a layout the model has never seen."""
    record, d, hh, mm = _base_record(rng)
    amount = round(rng.uniform(50, 30_000), 2)
    fee = float(rng.choice([0, 10])) if rng.random() < 0.4 else None
    record.update(
        doc_type="transfer",
        amount=amount,
        fee=fee,
        sender_name=_person(rng),
        sender_account=_masked_account(rng),
        receiver_name=_person(rng),
        receiver_account=_masked_account(rng),
        channel="ธนาคารทหารไทยธนชาต",
        reference=_reference(rng),
    )
    lines = [
        "ttb touch",
        "รายการโอนเงินสำเร็จ",
        f"{record['sender_name']} | {record['sender_account']}",
        "โอนไปยัง",
        f"{record['receiver_name']} | {record['receiver_account']}",
        f"{_render_amount(amount, rng)}",
    ]
    if fee is not None:
        lines.append(f"ค่าธรรมเนียมการโอน {_render_amount(fee, rng)}")
    lines += [
        f"{_render_date(d, rng)} · {_render_time(hh, mm, rng)}",
        f"เลขอ้างอิง {record['reference']}",
        "ธนาคารทหารไทยธนชาต",
    ]
    return "\n".join(lines), record


def _grab_receipt(rng: random.Random) -> tuple[str, dict]:
    """HELD OUT of training. A receipt whose total sits under a fare breakdown."""
    record, d, hh, mm = _base_record(rng)
    fare = round(rng.uniform(45, 320), 2)
    promo = round(rng.choice([0.0, 10.0, 20.0]), 2)
    total = round(fare - promo, 2)
    record.update(
        doc_type="receipt",
        amount=total,
        fee=None,
        receiver_name=None,
        channel="Grab",
        reference=_reference(rng),
    )
    text = "\n".join([
        "Grab",
        "ใบเสร็จการเดินทาง",
        f"{_render_date(d, rng)}  {_render_time(hh, mm, rng)}",
        f"ค่าโดยสาร  {fare:,.2f}",
        f"ส่วนลด  -{promo:,.2f}",
        f"ยอดชำระทั้งหมด  {_render_amount(total, rng)}",
        "ชำระผ่าน GrabPay",
        f"รหัสการเดินทาง {record['reference']}",
    ])
    return text, record


TEMPLATES = {
    "bank_transfer": (_bank_transfer, False),
    "promptpay": (_promptpay, False),
    "truemoney_topup": (_truemoney_topup, False),
    "wallet_transfer": (_wallet_transfer, False),
    "convenience_receipt": (_convenience_receipt, False),
    "cafe_receipt": (_cafe_receipt, False),
    "atm_transfer": (_atm_transfer, False),
    "mobile_banking_dense": (_mobile_banking_dense, False),
    "ttb_transfer": (_ttb_transfer, True),
    "grab_receipt": (_grab_receipt, True),
}

TRAIN_TEMPLATES = [name for name, (_, held) in TEMPLATES.items() if not held]
HELD_OUT_TEMPLATES = [name for name, (_, held) in TEMPLATES.items() if held]


def _add_noise(text: str, rng: random.Random) -> str:
    """OCR-ish damage that never touches a value.

    Only whitespace and label glyphs are disturbed. Corrupting a digit would
    make the label wrong rather than the input hard, and the whole point of a
    generated corpus is that the label cannot drift.
    """
    lines = text.split("\n")
    out = []
    for line in lines:
        if rng.random() < 0.25:
            line = line.replace(" ", "  ", 1)
        if rng.random() < 0.12:
            # A stray vertical bar: the slip's border, scanned. Deliberately not a
            # dot or a comma — those would land next to an amount and make the
            # printed value genuinely ambiguous, which the label could not follow.
            line = line.rstrip() + " |"
        if rng.random() < 0.10:
            line = line.replace("จำนวน", "จํานวน")  # the other Thai sara am, as real OCR emits
        out.append(line)
    return "\n".join(out)


def sample(rng: random.Random, template: str, noise: bool) -> Sample:
    fn, held = TEMPLATES[template]
    text, record = fn(rng)
    if noise:
        text = _add_noise(text, rng)
    return Sample(text=text, record=record, template=template, held_out=held)


def build_split(
    n: int, rng: random.Random, templates: list[str], noise_rate: float = 0.2
) -> list[Sample]:
    return [
        sample(rng, templates[i % len(templates)], noise=rng.random() < noise_rate)
        for i in range(n)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the slip corpus")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--train", type=int, default=1200)
    parser.add_argument("--valid", type=int, default=150)
    parser.add_argument("--test", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train = build_split(args.train, rng, TRAIN_TEMPLATES)
    valid = build_split(args.valid, rng, TRAIN_TEMPLATES)

    # The test split carries both held-out layouts as well, in a fixed proportion
    # so "unseen layout" is a large enough bucket to report on its own.
    seen = build_split(int(args.test * 0.7), rng, TRAIN_TEMPLATES)
    unseen = build_split(args.test - len(seen), rng, HELD_OUT_TEMPLATES)
    test = seen + unseen

    for name, rows in (("train", train), ("valid", valid)):
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for s in rows:
                f.write(json.dumps(chat_example(s.text, s.record), ensure_ascii=False) + "\n")
        print(f"  {path}  {len(rows)} rows")

    path = args.out / "test.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(test):
            f.write(json.dumps({
                "id": f"t{i:04d}",
                "text": s.text,
                "record": s.record,
                "template": s.template,
                "held_out": s.held_out,
            }, ensure_ascii=False) + "\n")
    print(f"  {path}  {len(test)} rows ({len(unseen)} on layouts held out of training)")


if __name__ == "__main__":
    main()
