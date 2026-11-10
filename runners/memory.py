#!/usr/bin/env python3
"""runner: memory -- does the model still know the brief after the context is
compactified?

This is the test no public benchmark runs, and it is the one that matters for
anybody actually working with an agent: contexts get compacted, and afterwards
the agent either still knows what it was doing or it quietly invents a new task.

Protocol, all inside one conversation:

1. A brief is delivered (session code, a numeric baseline, closed hypotheses).
2. The context is filled with unrelated material until it crosses the compaction
   threshold -- the filler is deliberately dull so the model cannot cheat by
   summarising it.
3. A compaction instruction is issued (stand-in for what an agent harness does
   automatically): "condense everything above into a short handover note".
4. The compacted note replaces the history, as a real harness would.
5. Three probes are asked, in order of difficulty:
     - recall the session code
     - recall the numeric baseline
     - answer a question that requires knowing which hypotheses are closed

Scoring: 1 point per probe, plus one point if the handover note itself preserved
both identifiers before the probes were asked. Reported as ``kept / 4``; a run
where the model could not produce a handover note at all is reported unscored.

Usage:
  python -m runners.memory --url http://127.0.0.1:1234 --model my-model
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string

from bench_common import Client, EngineError

COMPACT_PROMPT = (
    "Everything above will now be discarded. Write a handover note -- no more "
    "than 200 words -- containing anything you would need to continue this work "
    "afterward. Output only the note."
)

RECALL_PROMPT = (
    "From the work so far: what was the session code? If you do not know it, "
    "say UNKNOWN. Answer with the code only."
)

BASELINE_PROMPT = (
    "From the work so far: what was the numeric baseline (variant and value)? "
    "If you do not know it, say UNKNOWN. Answer with the value only."
)

CLOSED_PROMPT = (
    "From the work so far: is the 'cache stampede' hypothesis still open to "
    "investigate? Answer YES or NO only."
)


def filler(n_words: int, rng: random.Random) -> str:
    """Dull, low-information text. Nothing here is worth remembering."""
    words = []
    pool = ["record", "process", "system", "value", "result", "update", "item",
            "queue", "entry", "handle", "check", "amount"]
    for i in range(n_words):
        words.append(rng.choice(pool) if i % 3 else f"{rng.randint(1000, 9999)}")
    text = " ".join(words)
    return "\n".join(text[i:i + 400] for i in range(0, len(text), 400))


def make_brief(rng: random.Random) -> dict:
    code = f"{''.join(rng.choices(string.ascii_uppercase, k=3))}-{rng.randint(1000, 9999)}-{''.join(rng.choices(string.ascii_uppercase, k=5))}"
    variant = rng.choice(["A", "B", "C", "D"])
    value = f"{rng.randint(100, 900)}.{rng.randint(10, 99)}"
    hypotheses = ["cache stampede", "DNS failover", "scheduler drift",
                  "connection pooling", "index bloat"]
    closed = hypotheses[:3]
    return {"code": code, "variant": variant, "value": value,
            "closed": closed, "all": hypotheses,
            "text": (
                f"BRIEF\nSESSION CODE: {code}\n"
                f"BASELINE: variant {variant} = {value}\n"
                f"OPEN HYPOTHESES: {', '.join(hypotheses[3:])}\n"
                f"CLOSED HYPOTHESES (do not re-open): {', '.join(closed)}\n"
                f"GOAL: find why latency grew after the last deploy.\n"
            )}


def run_memory(client: Client, target_tokens: int, seed: int = 3) -> dict:
    rng = random.Random(seed)
    brief = make_brief(rng)

    messages = [{"role": "user", "content": brief["text"]},
                {"role": "assistant", "content": "Understood. I have the brief."}]
    # two filler turns so the "conversation" reads naturally
    per_turn = int(target_tokens * 8.9 / 2)
    for i in range(2):
        messages.append({"role": "user",
                         "content": filler(per_turn // 6, rng) +
                         f"\n\n(working notes, part {i + 1})"})
        messages.append({"role": "assistant", "content": "Noted."})

    messages.append({"role": "user", "content": COMPACT_PROMPT})
    handover = client.chat(messages, max_tokens=1024, temperature=0.0)

    low = handover["text"].lower()
    kept_code = brief["code"].lower() in low
    kept_value = brief["value"] in handover["text"]
    handover_prompt_tokens = handover.get("prompt_tokens")

    # a real harness replaces the history with the note
    after = [{"role": "system", "content": "Handover note from earlier work:\n" +
              (handover["text"] or "(empty)")}]

    probes = {}
    for name, prompt in (("code", RECALL_PROMPT), ("baseline", BASELINE_PROMPT),
                         ("closed", CLOSED_PROMPT)):
        r = client.chat(after + [{"role": "user", "content": prompt}],
                        max_tokens=48, temperature=0.0)
        probes[name] = {"answer": (r["text"] or "")[:80],
                        "finish_reason": r.get("finish_reason")}
        after.append({"role": "assistant", "content": r["text"] or ""})

    code_ok = brief["code"].lower() in probes["code"]["answer"].lower()
    base_ok = brief["value"] in probes["baseline"]["answer"]
    closed_ok = re.search(r"\bno\b", probes["closed"]["answer"], re.I) is not None

    kept = sum([kept_code, kept_value, code_ok, base_ok, closed_ok])
    scored = bool(handover["text"])
    return {
        "brief": brief,
        "handover": (handover["text"] or "")[:600],
        "handover_prompt_tokens": handover_prompt_tokens,
        "handover_kept_code": kept_code,
        "handover_kept_baseline": kept_value,
        "probes": probes,
        "probe_code_ok": code_ok,
        "probe_baseline_ok": base_ok,
        "probe_closed_ok": closed_ok,
        "kept": kept if scored else None,
        "kept_max": 5,
        "scored": scored,
        "thread_kept": bool(scored and kept >= 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1234")
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="unknown")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY"))
    ap.add_argument("--target-tokens", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results/memory.jsonl")
    args = ap.parse_args()

    client = Client(args.url, args.model, args.api_key)
    ok, msg = client.health()
    print(f"engine: {msg}")
    if not ok:
        raise SystemExit(2)
    print(f"filling context to ~{args.target_tokens} tokens, then compacting\n")

    os.makedirs("results", exist_ok=True)
    try:
        r = run_memory(client, args.target_tokens, args.seed)
    except EngineError as e:
        print(f"engine error: {e}")
        raise SystemExit(2) from None

    row = {"bench": "memory", "model": args.model, "quant": args.quant, **r}
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in row.items() if k != "brief"},
                  fh, ensure_ascii=False, indent=2)

    print(f"session code      {r['brief']['code']}")
    print(f"baseline          variant {r['brief']['variant']} = {r['brief']['value']}")
    print(f"compact at        {r['handover_prompt_tokens']} prompt tokens")
    print()
    print(f"note kept code    {r['handover_kept_code']}")
    print(f"note kept value   {r['handover_kept_baseline']}")
    print(f"probe: code       {r['probe_code_ok']}  -> {r['probes']['code']['answer'][:40]!r}")
    print(f"probe: baseline   {r['probe_baseline_ok']}  -> {r['probes']['baseline']['answer'][:40]!r}")
    print(f"probe: closed hyp {r['probe_closed_ok']}  -> {r['probes']['closed']['answer'][:40]!r}")
    print()
    print(f"KEPT {r['kept']}/{r['kept_max']}  thread_kept={r['thread_kept']}")
    print(f"->  {args.out}")


if __name__ == "__main__":
    main()
