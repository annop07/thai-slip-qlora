"""Score one or more prediction files against the test split.

Every configuration — base model, fine-tuned model, frontier model over an API —
writes the same prediction format and is scored by this one function. That is
what makes the rows of the benchmark comparable: no configuration gets its own
parser, its own leniency, or its own idea of what counts as correct.

    uv run python -m slipft.score results/*.predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from slipft.metrics import Report
from slipft.schema import FIELDS


def load_test(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {row["id"]: row for row in map(json.loads, f)}


def score_file(pred_path: Path, test: dict[str, dict]) -> dict:
    overall, seen, unseen = Report(), Report(), Report()
    label = pred_path.stem.replace(".predictions", "")

    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            item = test.get(row["id"])
            if item is None:
                raise SystemExit(f"{pred_path}: prediction {row['id']} is not in the test split")
            args = (
                item["record"],
                row.get("output", ""),
                row.get("latency_ms"),
                row.get("prompt_tokens"),
                row.get("completion_tokens"),
            )
            overall.add(*args)
            (unseen if item["held_out"] else seen).add(*args)

    missing = len(test) - overall.n
    return {
        "config": label,
        "model": next(iter({json.loads(l).get("model", "") for l in pred_path.open(encoding="utf-8")}), ""),
        "missing_predictions": missing,
        "overall": overall.summary(),
        "seen_layouts": seen.summary(),
        "unseen_layouts": unseen.summary(),
    }


def markdown(rows: list[dict]) -> str:
    out = [
        "| Configuration | Clean JSON | Field acc. | Exact record | Unseen layouts (field acc.) | Median latency |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        o, u = r["overall"], r["unseen_layouts"]
        out.append(
            f"| {r['config']} | {o['clean_json_rate']:.2f} | {o['field_accuracy']:.3f} | "
            f"{o['exact_record_rate']:.2f} | {u['field_accuracy']:.3f} | "
            f"{o['latency_ms_median']:.0f} ms |"
        )

    out += ["", "Per-field accuracy:", "",
            "| Field | " + " | ".join(r["config"] for r in rows) + " |",
            "| --- | " + " | ".join("---" for _ in rows) + " |"]
    for name in FIELDS:
        cells = " | ".join(f"{r['overall']['per_field_accuracy'][name]:.2f}" for r in rows)
        out.append(f"| {name} | {cells} |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score prediction files")
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/summary.json"))
    args = parser.parse_args()

    test = load_test(args.test)
    rows = [score_file(p, test) for p in args.predictions]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(markdown(rows))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
