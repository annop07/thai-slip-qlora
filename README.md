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
| Frontier-model baseline over the KKU proxy | ✅ `deepseek-v3.2`, 250/250 rows |
| QLoRA training run | ✅ 18.2 min on a free Colab T4, 4.1 GB peak |
| Base vs tuned, measured on the same GPU in the same session | ✅ 250 slips each |

**The 1.5B adapter matches the frontier model on this task** — and the held-out
layouts say where it does not.

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

## Results

250 held-out slips, greedy decoding, one scorer. Base and tuned are the same
weights before and after training, run on the same T4 in the same session.

| Configuration | Field acc. | Exact record | Clean JSON | Median latency | Completion tokens |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B base | 0.733 | 0.01 | 0.00 | 5,486 ms | 136 |
| **Qwen2.5-1.5B + QLoRA** | **0.995** | **0.96** | **1.00** | 6,571 ms | 106 |
| `deepseek-v3.2` over an API | 0.993 | 0.96 | 0.24 | 4,449 ms | 146 |

An 18-minute training run on a free GPU took a model that got **1 slip in 100**
completely right to one that gets **96 in 100** — level with a frontier model
that is hundreds of times its size, and ahead of it on output format.

Per field, where the 1.5B was losing:

| Field | base | tuned | `deepseek-v3.2` |
| --- | --- | --- | --- |
| doc_type | 0.92 | 1.00 | 1.00 |
| amount | 1.00 | 1.00 | 1.00 |
| fee | 0.77 | 1.00 | 1.00 |
| **date** | **0.10** | **0.97** | 0.98 |
| time | 0.97 | 1.00 | 1.00 |
| sender_name | 0.74 | 0.99 | 0.99 |
| sender_account | 0.70 | 0.98 | 0.99 |
| receiver_name | 0.60 | 1.00 | 0.99 |
| receiver_account | 0.60 | 1.00 | 0.99 |
| channel | 0.71 | 1.00 | 0.99 |
| reference | 0.96 | 1.00 | 1.00 |

**`date` at 0.10 is the whole story of the base model.** It reads the digits
correctly and then writes `2569-01-07`, or subtracts 543 from a slip that was
already Gregorian. The instruction to convert is right there in the prompt and
it does not follow it. After training: 0.97. That is the one field in this task
that is arithmetic rather than copying, and it is the one fine-tuning bought
outright.

**Format is trained, not requested.** Every base reply arrives wrapped in a
```json fence; so do three quarters of the frontier model's. The tuned model
emits a bare object 250 times out of 250. The prompt says "no markdown fence"
in all three cases — only the trained model actually obeys it, which is the
difference between a service that needs a fence-stripper forever and one that
does not.

## What the held-out layouts caught

Two slip layouts — `ttb touch` transfers and Grab receipts — were generated for
the test split only. Splitting the numbers by that boundary is where the honest
reading of "0.96" lives:

| | Seen layouts (175) | Unseen layouts (75) |
| --- | --- | --- |
| tuned, field accuracy | **1.000** | 0.982 |
| tuned, exact record | **1.000** | 0.853 |
| `deepseek-v3.2`, exact record | 0.960 | 0.947 |

On layouts it trained on, the adapter is **perfect — 175 of 175 slips, all
eleven fields**. On layouts it has never seen it drops to 0.853, while the
frontier model barely moves (0.960 → 0.947). Reported as a single number, the
fine-tune looks equal to the frontier model. Split by layout, it is better on
the familiar and worse on the unfamiliar, and that is the trade a fine-tune
actually makes.

**All 15 of the tuned model's remaining field errors are on one layout**, and
they have a shape:

```
อานนท์ ทองคำ | XXX-XXX5559 |
โอนไปยัง
อรทัย เกษมสุข | 651-1-6409-6
26,453.80                        <- the amount, alone on its line
17 ก.ย. 68 · 00:51               <- the date, unlabelled, directly below

truth  date = 2025-09-17
tuned  date = 2025-09-26         <- day borrowed from the line above
```

The Buddhist year is still converted correctly — `68` → 2025. What is lost is
*where the day lives*: in the eight training layouts the date follows a label
like `วันที่`, and here it does not, so the model reaches to the neighbouring
line. The account errors are the same failure in a different costume:
`368-4-9853-5` came back as `**4-9853-5`, the masking style of the *other*
account on the slip, which the model has learned is a thing accounts look like.

That is what over-specialisation looks like from the inside, and a benchmark
without held-out layouts would have printed 1.000 and said nothing about it.

## What it does not buy

**Speed, as measured here.** The tuned model's median latency is 6,571 ms
against 4,449 ms for an API call over the internet. Unbatched `model.generate`
on a T4 is a slow way to serve a 1.5B model — vLLM or llama.cpp on the same
weights would change this number completely, and the honest statement is that
this benchmark measured accuracy, not a serving stack.

What it does buy is everything the latency column is not: the slips never leave
the machine, the marginal cost per document is zero, the output format is
reliable enough to skip a parser, and the whole artifact is a **67 MB adapter**
over a 1 GB base — small enough to ship inside an app.

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

Reproducing the training run:

1. open [`notebooks/qlora_colab.ipynb`](notebooks/qlora_colab.ipynb) in Colab,
   **Runtime → Change runtime type → T4 GPU**
2. run it top to bottom — 18 min of training plus ~20 min for the two
   evaluation passes over 250 slips
3. drop `base.predictions.jsonl`, `tuned.predictions.jsonl` and `train_log.json`
   into `results/`, and score them with the same command as above

The run that produced the numbers above: 1,200 rows, 2 epochs, effective batch
8, lr 2e-4, LoRA `r=16` on all seven projections, loss on the answer only.
Final training loss 0.007, validation loss 0.0001 — a fit that tight is a
warning as much as a result, and the unseen-layout column is where it shows.

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

- **The tuned model is over-fitted to the layouts it saw** — 1.000 exact record
  on those, 0.853 on two it did not. Validation loss of 0.0001 says the same
  thing from the other end. More layouts, or aggressive augmentation of the
  ones there are, is the fix; a single headline number would have hidden it.
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
  in the same session and are comparable to each other only. None of the three
  is a serving benchmark — unbatched `model.generate` is the slowest reasonable
  way to run these weights.
- **The model reads text, not images.** A real pipeline needs OCR or a vision
  model in front of it, and its errors land here as garbled input. Nothing in
  this repo measures that half.
