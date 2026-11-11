# Credits and third-party notices

This benchmark set contains both original work and material derived from other
projects. Please read this file together with `LICENSE` (MIT, covering this
repository's own code).

## Derived material

### Quality suite — adapted from omar16100/llm-benchmark

`cases/quality.json` is adapted from the prompt set in
[omar16100/llm-benchmark](https://github.com/omar16100/llm-benchmark), which
publishes 26 prompts across six categories. We changed the scoring rule from
"Claude-as-judge" to deterministic checks, added a numeric-boundary guard and a
"failed unit test scores 1, not 0" rule, and run each case three times.

Upstream is MIT licensed. Its notice is reproduced below as the licence requires.

```
MIT License

Copyright (c) 2026 Omar Shabab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Long-context probe — idea credited to gkamradt/needle-in-a-haystack

The needle-in-a-haystack evaluation idea is popularised by
[gkamradt/needle-in-a-haystack](https://github.com/gkamradt/needle-in-a-haystack)
(MIT, Copyright (c) 2023 Greg Kamradt), and traces back to the "Lost in the
Middle" work on long-context retrieval.

**No code was copied from that project.** `runners/longctx.py` is an independent
implementation: it generates its own haystack text, plants its own needle, and
adds a measurement that upstream does not have -- the real `usage.prompt_tokens`
is compared against the requested size, and cells whose prompt was silently
truncated by the server are flagged and excluded from the pass count.

Because the technique is an evaluation method described openly in the literature
rather than a piece of code, the credit here is an acknowledgement of origin, not
a licence obligation.

## Original work

Everything below was written for this repository:

- `bench.py`, `bench_common.py`, `scorers.py`, `tests_scorers.py`
- `runners/speed.py` — throughput measurement with a stdout-noise guard
- `runners/longctx.py` — including the truncation guard
- `runners/session.py` and `runners/make_session_data.py` — the 26-file project,
  the seeded defects, and the review scoring
- `runners/memory.py` — compaction and handover recall
- `runners/totals.py` — arithmetic self-consistency
- the scoring rules described in `README.md`, including the numeric-boundary
  guard, the "failed test scores 1" rule, and the exclusion of judge-only cases

## Reporting an attribution problem

If you believe something here is credited incorrectly or used without
permission, open an issue with the file name and we will fix or remove it.
