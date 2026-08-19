"""The 'just call a big model' row of the benchmark.

This is the configuration a fine-tune has to beat to be worth doing: no
training, no GPU, one API call per slip. It is also the only row whose latency
is not measured on the same hardware as the others — it includes the network
and someone else's queue — so latency is reported for it but not compared.

    uv run python -m slipft.api_baseline --model qwen3-next-80b-a3b-instruct
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from slipft.prompt import SYSTEM, user_message


RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class QuotaExhausted(Exception):
    """The proxy has cut this model off for the day. Retrying cannot help."""


async def one(client: httpx.AsyncClient, model: str, row: dict, sem: asyncio.Semaphore,
              stop: asyncio.Event, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(row["text"])},
        ],
        # Greedy: the benchmark is about extraction accuracy, and sampling would
        # make the same run return different numbers.
        "temperature": 0,
        # 1200, for a JSON answer that never exceeds ~200 tokens. `gemini-3.6-flash`
        # spends its budget on hidden reasoning before the first visible character:
        # at a 400-token cap it returned 396 completion tokens and 36 characters of
        # truncated JSON on every single slip, scoring 0.000 — which would have read
        # as "the model cannot do the task" rather than "the cap was in the wrong place".
        "max_tokens": max_tokens,
    }

    def fail(message: str, elapsed: float) -> dict:
        return {"id": row["id"], "model": model, "output": "", "error": message[:200],
                "latency_ms": elapsed * 1000}

    async with sem:
        started = time.perf_counter()
        if stop.is_set():
            return fail("skipped: daily limit already reached", time.perf_counter() - started)

        last_error = ""
        for attempt in range(3):
            try:
                r = await client.post("/chat/completions", json=body)
                # The proxy answers an exhausted per-model daily quota with 401 and
                # {"error": "This model reached daily limit."} — not 429. Retrying
                # that burns the rest of the run against a wall, so the whole run
                # stops instead and the file says why.
                if r.status_code == 401 and "daily limit" in r.text.lower():
                    stop.set()
                    raise QuotaExhausted(r.text.strip())
                r.raise_for_status()
                data = r.json()
                break
            except QuotaExhausted as e:
                return fail(f"daily limit reached: {e}", time.perf_counter() - started)
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:120]}"
                if e.response.status_code not in RETRYABLE:
                    return fail(last_error, time.perf_counter() - started)
            except httpx.HTTPError as e:  # timeouts, connection resets
                last_error = f"{type(e).__name__}: {e}"
            if attempt < 2:
                await asyncio.sleep(1.5 * (2**attempt))
        else:
            return fail(last_error or "exhausted retries", time.perf_counter() - started)

        latency_ms = (time.perf_counter() - started) * 1000

    choices = data.get("choices") or []
    if not choices:
        return fail(f"no choices in response: {json.dumps(data)[:120]}", latency_ms / 1000)

    usage = data.get("usage") or {}
    return {
        "id": row["id"],
        "model": model,
        # `content` can be null when a model spends its whole budget on reasoning;
        # that is an empty answer, which scores zero, not a crash.
        "output": (choices[0].get("message", {}).get("content") or "").strip(),
        "latency_ms": latency_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


async def run(model: str, rows: list[dict], concurrency: int, out: Path,
              max_tokens: int, allow_partial: bool = False) -> None:
    load_dotenv()
    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY")
    if not base_url or not key:
        raise SystemExit("set OPENAI_BASE_URL and OPENAI_API_KEY (see .env.example)")

    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    async with httpx.AsyncClient(
        base_url=base_url, headers={"Authorization": f"Bearer {key}"}, timeout=120
    ) as client:
        started = time.perf_counter()
        results = await asyncio.gather(*(one(client, model, r, sem, stop, max_tokens) for r in rows))
    elapsed = time.perf_counter() - started

    failed = sum(1 for r in results if r.get("error"))

    # Write beside the target and rename only once the run is worth keeping. An
    # earlier version wrote straight to `out`, and a run that died against a spent
    # quota overwrote a good result file with 250 rows of nothing — a whole
    # baseline lost to a rerun that could never have succeeded.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Any hole at all keeps the file out of `results/`. A run that lost 66 of 250
    # rows to a spent quota is not a measurement of the model, and the previous
    # rule — replace unless *everything* failed — let exactly that overwrite a
    # complete run. Re-run it, or pass --allow-partial deliberately.
    if failed and not allow_partial:
        print(f"{model}: {failed}/{len(results)} rows failed — {out} left as it was, "
              f"dump in {tmp}")
        if stop.is_set():
            print("  this model's daily quota on the proxy is spent; try again tomorrow.")
        return

    tmp.replace(out)
    print(f"{model}: {len(results)} rows in {elapsed:.0f}s, {failed} failed -> {out}")
    if stop.is_set():
        print("  stopped early: this model's daily quota on the proxy is spent. "
              "Re-run tomorrow — a part-empty file would score the quota, not the model.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the API baseline over the test split")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--allow-partial", action="store_true",
                        help="write the file even though some rows failed")
    args = parser.parse_args()

    with args.test.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]

    out = args.out or Path("results") / f"api-{args.model}.predictions.jsonl"
    asyncio.run(run(args.model, rows, args.concurrency, out, args.max_tokens,
                    args.allow_partial))


if __name__ == "__main__":
    main()
