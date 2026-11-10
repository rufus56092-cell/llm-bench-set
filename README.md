# LLM Bench Set

A small, dependency-free benchmark set for local LLMs. Run the same tests on
your laptop, your desktop, a Spark, or a Mac, and get numbers you can put side
by side.

Built out of frustration with two things: benchmarks that only say *"it runs"*,
and stacks that happily accept a 100K prompt while quietly cutting it to 4K.

```
git clone https://github.com/rufus56092-cell/llm-bench-set
cd llm-bench-set
python bench.py --url http://127.0.0.1:1234 --model your-model --quant Q4_K_M
```

That's it. Requirements: Python 3.10+. No pip install, no API keys, no telemetry.

## What it measures

| Bench | Question it answers | Runtime* |
|---|---|---|
| `speed` | How fast is prefill, decode, and time-to-first-token? | ~2 min |
| `quality` | Does it still get answers right? 78 tasks, 6 categories | ~30 min |
| `longctx` | Does it really read a long prompt, or silently truncate? | ~15 min |
| `session` | Can it work through ~95 files without losing the plot? | ~35 min |
| `memory` | Does it remember the brief after the context is compacted? | ~25 min |
| `totals` | Does it keep its own numbers straight across a report? | ~5 min |

\* measured on a 177B MoE at 2 bits on one 22 GB GPU. A fast server is quicker,
a CPU box is much slower.

Run a subset:

```bash
python bench.py --url ... --model ... --only speed,quality
python bench.py --url ... --model ... --only quality --only-cases C1,M1,R1
python bench.py --url ... --model ... --only longctx --sizes 8000,32000
```

Compare runs from different machines:

```bash
python bench.py --compare results/*_run.json
```

## Design rules

These are the choices that make numbers comparable between machines. They are
not incidental.

**1. Context size is read from the server, never guessed.**
`longctx` reports `usage.prompt_tokens` of the actual response and compares it
to what was requested. A cell whose real prompt came back far below target is
flagged `TRUNCATED` and does not count as a pass. This is the single most
important guard: a stack that silently truncates will otherwise look perfect.

**2. Throughput comes from the server's own timings when it has them.**
llama.cpp and LM Studio report `prompt_per_second` / `predicted_per_second`.
When those exist they are used; otherwise speed is derived from `usage` and wall
clock and the row is marked `speed_source: client`. Client wall time is never
reported as prefill — it includes generation.

**3. Scoring is deterministic wherever possible.**
Exact match, unit tests, constraint checks. Cases with only a subjective
criterion are excluded from the score rather than judged by a model, so the
number does not depend on which judge you happen to have.

**4. A short numeric answer must match as a number.**
`1/2` does not match inside `1/20`. A correct answer buried in prose scores 3,
not 5 — right idea, wrong format.

**5. A failed unit test scores 1, not 0.**
Producing runnable Python that fails is a different outcome from producing
nothing. Collapsing them hides real differences between models.

## Results format

Every run writes `results/<bench>.jsonl` (one line per task) and
`results/<bench>_summary.json` (aggregate). `bench.py` merges them into
`results/<model>__<quant>_run.json`.

```json
{
  "model": "qwen3.8-flash-next-iq2_xs",
  "quant": "IQ2_XS",
  "benchmarks": {
    "speed":   {"decode_tok_s_mean": 55.6, "prefill_tok_s_mean": 691.0},
    "quality": {"overall": 4.72, "by_category": {"math": 5.0, "coding": 4.47}},
    "longctx": {"recall": 1.0, "truncated_cells": 0}
  }
}
```

Flat JSON on purpose: `jq`, a spreadsheet, or a 10-line script can diff two runs.

## Reference numbers

One RTX 2080 Ti (22 GB), 128 GB RAM, Qwen3.8-Flash-Next 177B MoE at IQ2_XS,
131072-token window. 21.9 of 22 GB used — no headroom, so anything else on the
GPU will slow it down.

| Metric | Value |
|---|---|
| decode | 55.6 tok/s mean, 67.2 peak |
| prefill | 691 tok/s mean at 55K prompt |
| quality | 4.72 / 5 across 78 tasks |
| longctx | 12/12 cells, verified to 101 339 tokens |

Your hardware will differ. That is the point of running it.

## Credits

The 78-case quality suite is adapted from
[omar16100/llm-benchmark](https://github.com/omar16100/llm-benchmark); the
needle-in-a-haystack idea is the standard one popularised by
[gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack).
The runner code, the truncation guard, the retrieval-recall test, and the
session/memory benches are ours.

License: MIT.
