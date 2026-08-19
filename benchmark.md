# Benchmark

Reproduce with:

```bash
uv run python -m slipft.generate                              # deterministic, seed 20260819
uv run python -m slipft.api_baseline --model deepseek-v3.2    # 250 calls, ~4 min
uv run python -m slipft.score results/*.predictions.jsonl
```

Every configuration is scored by `slipft.score` from a prediction file in one
format. Nothing in the scorer knows which model produced a file.

## Method

**Test set.** 250 slips held out of training, from the same generator: 175 on
the eight layouts that appear in training, 75 on two that never do
(`ttb_transfer`, `grab_receipt`). They are reported separately so that "learned
to extract" can be told apart from "learned nine templates".

**Prompt.** One system prompt, in [`src/slipft/prompt.py`](src/slipft/prompt.py),
used by the training data, the base model, the tuned model and the API models.
The usual way to flatter a fine-tune is to train it on a short prompt and
compare it against a long one; here the prompt is a constant.

**Decoding.** Greedy everywhere — `temperature: 0` over the API,
`do_sample=False` in the notebook. A benchmark that resamples is a benchmark
that cannot be re-run.

**Scoring.** Eleven fields, each scored independently, `null` included:

| Field | Compared as |
| --- | --- |
| `amount`, `fee` | numbers, tolerance 0.005 — `1234.5` == `"1,234.50"` |
| `time` | normalised to `HH:MM` — `14.32 น.` == `14:32` |
| `channel` | through an alias table — `SCB` == `ธนาคารไทยพาณิชย์`, `GrabPay` == `Grab` |
| everything else | exact string after stripping, masking characters included |

A reply that does not parse scores zero on all eleven fields; there is no
partial credit for prose that happens to contain the right number. `null` is a
prediction: answering `fee: 0` on a slip that prints no fee is wrong, and so is
answering `fee: null` on one that prints 10.00.

**Reported numbers.** `field_accuracy` is correct fields over 11 × 250.
`exact_record_rate` is the share of slips where all eleven are right — the
number that matters if the output feeds a database without review.
`clean_json_rate` is the share of replies that parse with no recovery at all
(no fence, no prose); accuracy does not depend on it, but a service does.

## Results

### `deepseek-v3.2` over the KKU proxy — 2026-08-19

250/250 rows returned, 0 failures, greedy, `max_tokens` 1200.

| | Overall | Seen layouts (175) | Unseen layouts (75) |
| --- | --- | --- | --- |
| Field accuracy | 0.993 | 0.993 | 0.995 |
| Exact record | 0.956 | 0.960 | 0.947 |
| Clean JSON | 0.24 | 0.23 | 0.25 |
| Schema valid | 1.00 | 1.00 | 1.00 |
| Median latency | 4,449 ms | 4,546 ms | 4,147 ms |
| p90 latency | 8,080 ms | 9,069 ms | 5,867 ms |
| Prompt / completion tokens | 432 / 146 | — | 429 / 116 |

Per field:

| Field | Accuracy |
| --- | --- |
| doc_type | 1.00 |
| amount | 1.00 |
| fee | 1.00 |
| date | 0.98 |
| time | 1.00 |
| sender_name | 0.99 |
| sender_account | 0.99 |
| receiver_name | 0.99 |
| receiver_account | 0.99 |
| channel | 0.99 |
| reference | 1.00 |

**Unseen layouts are not harder for a model this size** — 0.995 against 0.993.
That is expected of a frontier model and is exactly the comparison to watch
when the 1.5B rows arrive: a small model that has memorised eight templates
will show the gap here first.

**`date` is the only column that is not saturated**, and every miss is the
Buddhist year: `2568` read as `2026` instead of `2025`, or a two-digit `69` left
as `2069`. It is the one part of this task that is arithmetic rather than
copying, and it is where a fine-tune has something to learn.

### Qwen2.5-1.5B-Instruct, base and QLoRA — Colab T4, 2026-08-19

Both rows come from one session of
[`notebooks/qlora_colab.ipynb`](notebooks/qlora_colab.ipynb): the base model is
evaluated before the adapter is attached, the tuned model after, same GPU, same
prompt, same 250 slips.

Training: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`, LoRA `r=16` `alpha=16` on
`q,k,v,o,gate,up,down`, 1,200 rows, 2 epochs, effective batch 8, lr 2e-4 cosine,
loss on the answer span only. **18.2 minutes, 4.12 GB peak, train loss 0.0069,
validation loss 0.0001.**

| | base | tuned | `deepseek-v3.2` |
| --- | --- | --- | --- |
| Field accuracy | 0.733 | **0.995** | 0.993 |
| Exact record | 0.012 | **0.956** | 0.956 |
| Clean JSON | 0.00 | **1.00** | 0.24 |
| Schema valid | 0.98 | 1.00 | 1.00 |
| Parsed at all | 1.00 | 1.00 | 1.00 |
| Median latency | 5,486 ms | 6,571 ms | 4,449 ms |
| p90 latency | 6,500 ms | 7,447 ms | 8,080 ms |
| Completion tokens | 136 | 106 | 146 |

Per field:

| Field | base | tuned | `deepseek-v3.2` |
| --- | --- | --- | --- |
| doc_type | 0.92 | 1.00 | 1.00 |
| amount | 1.00 | 1.00 | 1.00 |
| fee | 0.77 | 1.00 | 1.00 |
| date | 0.10 | 0.97 | 0.98 |
| time | 0.97 | 1.00 | 1.00 |
| sender_name | 0.74 | 0.99 | 0.99 |
| sender_account | 0.70 | 0.98 | 0.99 |
| receiver_name | 0.60 | 1.00 | 0.99 |
| receiver_account | 0.60 | 1.00 | 0.99 |
| channel | 0.71 | 1.00 | 0.99 |
| reference | 0.96 | 1.00 | 1.00 |

**The base model can read and cannot convert.** `amount` 1.00 with `date` 0.10
is not a model that fails to see the date — it copies the printed digits and
returns `2569-01-07`, or subtracts 543 from the one slip in six that was already
Gregorian. The prompt states the rule; the base model does not apply it. This
single field is 60% of the gap between 0.733 and 0.995.

**Its other weakness is knowing which name is whose.** `receiver_name` and
`receiver_account` at 0.60 against `sender_*` at 0.70–0.74: on layouts where the
two parties are two similar lines, the base model swaps them or drops one. Both
go to 0.99–1.00 after training on 1,200 examples of the distinction.

### Split by layout

| | Seen layouts (175) | Unseen layouts (75) |
| --- | --- | --- |
| base — field acc. / exact | 0.718 / 0.017 | 0.766 / 0.000 |
| tuned — field acc. / exact | **1.000 / 1.000** | 0.982 / 0.853 |
| `deepseek-v3.2` — field acc. / exact | 0.993 / 0.960 | 0.995 / 0.947 |

The frontier model is flat across the boundary — 0.960 and 0.947 — because
there is no boundary for a model that never saw either set. The tuned model is
**perfect on the eight layouts it trained on and loses 15 percentage points of
exact record on the two it did not**. Both facts are true; only one of them
survives being reported as a single number.

All 15 of the tuned model's wrong fields are on `ttb_transfer`, none on
`grab_receipt`, and they are positional rather than semantic:

| Count | Field | What happened |
| --- | --- | --- |
| 8 | `date` | day-of-month taken from the amount on the line above — `26,453.80` above `17 ก.ย. 68` came back as `2025-09-26`. The Buddhist year was still converted correctly. |
| 4 | `sender_account` | `368-4-9853-5` returned as `**4-9853-5`, borrowing the masking style of the *other* account on the same slip |
| 2 | `sender_name` | taken from the wrong side of a `|`-separated line |
| 1 | `receiver_account` | same as above |

In the eight training layouts a date follows a label such as `วันที่`; in
`ttb touch` it does not. The adapter learned the arithmetic and lost the search.
That is the specific shape of over-specialisation, and the held-out layouts are
the only reason it is visible.

## What scoring decisions cost, measured

Two of them moved the numbers more than any model choice did.

**The token cap.** `gemini-3.6-flash` scored **0.000** on everything — not one
parseable reply out of 250. It spends its budget on hidden reasoning before the
first visible character: 396 completion tokens against a 400-token cap, leaving
a median of 36 characters of truncated JSON. Raising the cap to 1,200 is the
whole fix. A benchmark that had published that 0.000 would have been reporting
its own configuration error as a property of the model.

**The naming convention.** Scored by exact string, `deepseek-v3.2` gets 0.957
field accuracy and 0.58 exact records. The same predictions scored through the
alias table get 0.993 and 0.96. The 103 disagreements were:

| Count | Label | Model said |
| --- | --- | --- |
| 33 | ธนาคารทหารไทยธนชาต | `ttb`, `ttb touch` |
| 39 | TrueMoney Wallet | `TrueMoney` |
| 31 | ธนาคาร… (full Thai name) | `SCB`, `KTB`, `GSB`, or the name without the ธนาคาร prefix |

None is a reading error, and the fix has to be in the scorer rather than the
prompt: the alternative is a benchmark where a fine-tuned model wins by
learning which of several correct names this corpus happens to prefer.

## Caveats

- **The corpus is synthetic.** The labels are correct by construction, and the
  slips are as realistic as a generator makes them. Numbers here transfer to
  real slips only as far as the templates do.
- **250 slips, 2,750 field judgements.** A one-question difference in the
  exact-record column is 0.004; differences under ~0.02 are noise at this size.
- **Only one API model completed a full run.** Four others hit the proxy's
  per-model daily quota during development, which the runner now detects rather
  than writing a file full of holes.
- **The tuned model's validation loss is 0.0001.** On a generated corpus that is
  a warning, not an achievement: the task has one right answer per slip and only
  eight layouts to learn it from. The unseen-layout split is the only part of
  this report that number does not flatter.
- **The API row's latency includes the network.** It is reported because a
  service pays it, not because it is comparable to a number measured on a T4.
  None of the three rows is a serving benchmark — unbatched `model.generate` is
  the slowest reasonable way to run a 1.5B model, and vLLM or llama.cpp on the
  same weights would move the local numbers by a lot.
