# Benchmark results

Measured on one machine, published as a reference point. Your numbers will
differ -- that is the point of running the set yourself.

**Machine:** 1× RTX 2080 Ti (22 GB, undervolted to 180 W), AMD EPYC 7D12,
128 GB RAM, model fully resident in VRAM (21.9 of 22 GB used).

**Model:** Qwen3.8-Flash-Next, 177B MoE (125B + 51B n-gram table), GSQ-RCO
IQ2_XS quantisation, 131072-token window, int8 KV cache.

## Summary

| Bench | Result |
|---|---|
| `speed` | 55.6 tok/s decode, 691 tok/s prefill (at 55K prompt) |
| `quality` | 3.76 / 5 across 75 scored tasks |
| `longctx` | 9/9 cells found, 0 truncated, verified to 64 801 tokens |
| `session` | 12/12 files read, 5/12 defects, brief recalled |
| `memory` | 5/5 kept through compaction at 70 154 prompt tokens |
| `totals` | arithmetic correct, 4/4 sections |

## Quality by category

| Category | Score | Tasks |
|---|---|---|
| instruction following | 4.88 | 12 |
| coding | 4.73 | 15 |
| creative | 4.25 | 12 |
| reasoning | 3.25 | 12 |
| math | 3.00 | 12 |
| tool use | 2.19 | 12 |
| **overall** | **3.76** | **75** |

Three runs per case, 78 runs total; three unscored (two truncated at the token
cap, one produced no text).

### Run-to-run variance

These numbers move between runs, and the direction matters:

- a repeat of the same quality suite scored 3.85 over 71 tasks instead of 3.76
  over 75. The higher average is **not** a better run: four tool-use cases came
  back with an empty answer and were excluded rather than scored zero, so the
  weakest category shrank from 12 tasks to 8 and stopped pulling the mean down.
  Treat a rising score with a falling task count as suspicious, not as progress.
- the session bench found 7 defects on one run, 6 on another and 5 on the third,
  with the same code and prompt. The misses differ per run, so read that bench as
  "5-7 of 12" rather than a fixed figure.

A single run of anything here is an anecdote. Two runs that disagree by more than
the variance above mean the configuration changed, not that the model got better.

The two weak categories are informative rather than surprising. **Math** at 3.00
is a formatting problem more than a reasoning one: the model solves the problem
and writes the working, then buries the answer in prose, which scores 3 instead
of 5 under the "answer in prose" rule. **Tool use** at 2.19 is a real weakness --
it calls the right tool but gets the arguments wrong about half the time, and
one case it never answered at all.

## Long context

The important number is not the recall, it is `truncated_cells`. A stack that
silently cuts the prompt produces a perfect score on a model that cannot read.

| Requested | Real prompt | Depth 5% | 50% | 95% |
|---|---|---|---|---|
| 8K | 8 132 | found | found | found |
| 32K | 32 426 | found | found | found |
| 64K | 64 801 | found | found | found |

9/9 cells, 0 truncated. Prefill on a 64K prompt costs 78–147 s, which is the
real price of long context on this machine -- generation is fast, the wait
before the first token is not. Prefill throughput falls from 693 tok/s at 8K to
450 tok/s at 64K.

A previous run on a differently-configured stack (no `cache_prompt`, different
KV quantisation) also cleared 100K at 101 339 tokens, 12/12 cells. That number is
not reproduced here because the run in this table used the 64K ceiling of the
current configuration.

## Session

26 files (12 source modules, 8 documents, 5 incident notes, 1 brief), 30-turn
budget. The model read all 12 source modules, wrote a 5 KB findings file, and
reproduced the session code from the brief without being prompted.

Five of twelve seeded defects found. Across three runs the count was 5, 6 and 7
with different misses each time. The misses are consistent in kind: it caught the
structural problems (non-atomic write, uncapped page size, mutation during
iteration) and missed the subtler ones (timing-safe comparison, unsigned
timestamp, lowercase mismatch). That split is more useful than a single number --
it says the model reads code well and reviews security poorly.

Its issue count (27 claimed vs 32 counted) was inside the tolerance, so the
self-consistency check passed, but the gap suggests it was counting lines rather
than defects.

## Memory through compaction

Context filled to 70 154 prompt tokens, then compacted under an instruction to
write a handover note. The note itself preserved both the session code and the
numeric baseline, and all three probes answered correctly afterwards, including
the one requiring it to know which hypotheses were closed.

5/5. This is the bench with the least prior art and, in practice, the one that
predicts whether an agent stays usable across a long session.

## Totals

27 items across four categories. The model reported the correct total (27), named
the correct largest category, and got all four section counts right. No
drift between the section numbers and the summary.

## Reproducing

```bash
python bench.py --url http://127.0.0.1:1234 --model qwen3.8-flash-next-iq2_xs --quant IQ2_XS
```

Raw per-task records are in `results/*.jsonl` on the machine that ran it; they
are not committed here.
