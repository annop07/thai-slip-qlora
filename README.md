# QLoRA — Thai payment-slip extraction on a 1.5B model

> **Week 7 bootcamp deliverable** — fine-tune a small open model to read Thai
> transfer slips and receipts into a fixed JSON schema, then measure whether it
> is worth doing: the same model before and after, and a frontier model over an
> API, scored by one scorer on one held-out set.

The question is not "can an LLM read a slip" — a large one can, and the number
below says so. It is whether a **1.5B model you run yourself** can do it well
enough that the financial documents never leave the machine, the answer arrives
in a fraction of the time, and the per-slip cost is zero.

## Status

| Piece | State |
| --- | --- |
| Corpus generator, 10 slip layouts, 2 held out of training | ✅ 1,200 / 150 / 250 rows, reproducible from a seed |
| Schema, prompt, scorer, unit tests | ✅ 39 tests |
| Frontier-model baseline over the KKU proxy | ✅ measured — `deepseek-v3.2`, 250/250 rows |
| QLoRA training + base/tuned evaluation notebook | ✅ written, ▶︎ **not yet run** (Colab T4) |
| Base vs tuned rows of the benchmark | ⬜ produced by that notebook run |

Everything except the training run is measured, and the training run is one
notebook away. The reason it is not in this repo yet is stated plainly below.

## Where the training runs, and why not here

**Not on the API key.** The KKU IntelSphere endpoint is inference only:

```
GET https://gen.ai.kku.ac.th/api/v1/fine_tuning/jobs  ->  404 Cannot GET
```

There is no endpoint to upload a dataset to. An API key answers questions with
somebody else's weights; it cannot produce new ones.

**Not on this laptop.** QLoRA quantises the frozen base with bitsandbytes' NF4
kernels, which are CUDA-only — an Apple M2 cannot run them at all, and the
Apple-native alternative (MLX LoRA over a 4-bit base) would pin a fanless
machine at full load for 30–45 minutes to train a model this size.

**So: a free Colab T4**, which is where QLoRA actually runs as written in the
brief — Unsloth, bitsandbytes NF4, `train_on_responses_only`. The notebook
evaluates the base model *before* attaching the adapter and the tuned model
after, on the same GPU in the same session, so the two rows differ by the
weights and nothing else.

## The task

Eleven fields out of a slip that was never designed to be machine-read:

```
ธนาคารกรุงไทย                          {"doc_type":"transfer",
โอนเงินสำเร็จ                            "amount":46962.28,
วันที่ 7 ม.ค. 69  เวลา 10:04 น.    ->     "fee":null,
จาก พรทิพย์ ทองคำ                        "date":"2026-01-07",
XXX-XXX8373                              "time":"10:04",
ไปยัง กิตติศักดิ์ วงศ์วิวัฒน์              "sender_account":"XXX-XXX8373",
x7810                                    "receiver_account":"x7810",
จำนวนเงิน ฿46,962.28                     "channel":"ธนาคารกรุงไทย",
รหัสอ้างอิง: 256584563536782              "reference":"256584563536782", ...}
```

The hard part is not finding the numbers, it is normalising them. `69` is a
two-digit **Buddhist** year: 2569 minus 543 is 2026, and getting that wrong is
the single most common error in every configuration measured so far. Amounts
lose their commas and their ฿; account masks keep theirs exactly.

## What it demonstrates

| Skill | Where it lives |
| --- | --- |
| Instruction dataset built from a schema, not scraped | [`src/slipft/generate.py`](src/slipft/generate.py) |
| One schema driving generation, prompting, validation and scoring | [`src/slipft/schema.py`](src/slipft/schema.py) |
| A prompt every configuration shares, so the only variable is the weights | [`src/slipft/prompt.py`](src/slipft/prompt.py) |
| Field-level scoring where a missing value is a prediction | [`src/slipft/metrics.py`](src/slipft/metrics.py) |
| One scorer for local models and API models alike | [`src/slipft/score.py`](src/slipft/score.py) |
| Async API baseline with retry, quota detection and atomic writes | [`src/slipft/api_baseline.py`](src/slipft/api_baseline.py) |
| QLoRA on a 4-bit base, loss on the answer only | [`notebooks/qlora_colab.ipynb`](notebooks/qlora_colab.ipynb) |
| Held-out layouts that separate "learned the task" from "memorised the format" | `ttb_transfer`, `grab_receipt` — test split only |
| Tests that pin the labels rather than the code paths | [`tests/`](tests/) |

## Results so far

`deepseek-v3.2` over the KKU proxy, 250 held-out slips, greedy, 2026-08-19:

| Configuration | Field acc. | Exact record | Unseen layouts | Clean JSON | Median latency | Completion tokens |
| --- | --- | --- | --- | --- | --- | --- |
| `deepseek-v3.2` (API) | **0.993** | **0.96** | 0.995 | 0.24 | 4,449 ms | 146 |
| Qwen2.5-1.5B base | — | — | — | — | — | — |
| Qwen2.5-1.5B + QLoRA | — | — | — | — | — | — |

Per field, the frontier model misses almost nothing: `date` 0.98 is its weakest
column and everything else is 0.99 or 1.00. **That is the bar**, and it sets up
the actual question — a 1.5B adapter does not have to be better, it has to be
close enough that 4.4 seconds and a round trip to a server become the
difference that matters.

Two things the API row already showed:

**`clean_json_rate` of 0.24 is a formatting habit, not a failure.** Three
quarters of the replies arrive wrapped in a ```json fence despite the prompt
saying not to. The scorer digs the object out, so accuracy is unaffected — but
a service would have to strip fences forever. This is the column a fine-tune
should take to 1.00, because the format is trained rather than requested.

**Naming conventions were worth 38 points of exact-record.** Scored by exact
string, the same predictions give 0.957 field accuracy and **0.58** exact
records; the model answered "SCB" where the corpus says ธนาคารไทยพาณิชย์,
"TrueMoney" for "TrueMoney Wallet", "ttb touch" for the bank's full Thai name.
None of those is a reading error. `channel` is now scored through an alias
table ([`metrics.py`](src/slipft/metrics.py)), which lifts it from 0.59 to 0.99
and the record rate to 0.96. Left as-is it would have flattered the fine-tuned
model most of all: a model trained on 1,200 of these labels learns the house
style in the first fifty examples, and the gap would have been the labels, not
the extraction.

## Quick start

```bash
uv run python -m slipft.generate            # 1,200 / 150 / 250 rows from a seed
uv run --extra dev pytest                   # 39 tests
```

Baseline against a hosted model (needs `.env`, see `.env.example`):

```bash
uv run python -m slipft.api_baseline --model deepseek-v3.2
uv run python -m slipft.score results/*.predictions.jsonl
```

Then the training run:

1. open [`notebooks/qlora_colab.ipynb`](notebooks/qlora_colab.ipynb) in Colab,
   **Runtime → Change runtime type → T4 GPU**
2. run it top to bottom — ~15 min of training, ~20 min for the two evaluation
   passes over 250 slips
3. drop `base.predictions.jsonl`, `tuned.predictions.jsonl` and `train_log.json`
   into `results/`, and score them with the same command as above

The notebook produces predictions only. Scoring stays here so that the local
models and the API models are graded by the same code — a benchmark where each
row brings its own parser is not a benchmark.

## The corpus is generated, and that is a limitation with a fence around it

Real slips carry real account numbers and real names, and hand-labelling a
thousand of them was not going to happen in a week. Generating them buys a
ground truth that is **correct by construction** — the value is sampled first
and the slip is rendered around it, so a disagreement is always the model's.

What it costs is realism, and every number here should be read as "on slips
that look like these". Three things keep that from being a free pass:

- **Two layouts never appear in training.** `ttb_transfer` and `grab_receipt`
  are generated for the test split only, and are reported as their own column.
- **Values are drawn independently of layout.** Date formats are not tied to
  banks, so a model cannot shortcut the Buddhist-year conversion by recognising
  which bank printed the slip. One format in six is Gregorian, so "always
  subtract 543" is wrong too.
- **20% of rows carry OCR-ish damage** — doubled spaces, a stray border bar, the
  other Unicode form of sara am. Never a digit: corrupting a value would make
  the label wrong instead of the input hard.

## Known limits

- **The headline rows are not filled in yet.** Base and tuned come from the
  Colab run, and this README will carry whatever they turn out to be, including
  a fine-tune that fails to reach the API baseline.
- **Generated data, self-authored schema.** See above. An evaluation on 50
  photographed real slips would be worth more than all 250 of these, and is the
  obvious next step.
- **One API baseline, not four.** `qwen3-next-80b-a3b-instruct`,
  `claude-haiku-4.5`, `gemini-3.6-flash` and `llama-4-scout` all hit the proxy's
  per-model daily quota during development. The runner detects that case now and
  refuses to write a part-empty file; re-running them is one command each.
- **`gemini-3.6-flash` needs its own note.** At a 400-token cap it scored
  **0.000** — it spends ~396 completion tokens on hidden reasoning before the
  first visible character, so every reply was truncated JSON. That is a cap in
  the wrong place, not a model that cannot read Thai. The default is now 1,200.
- **Latency across rows is not apples to apples.** The API row includes the
  network and someone else's queue; base and tuned are measured on the same T4
  in the same session and are comparable to each other only.
