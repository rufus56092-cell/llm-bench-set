#!/usr/bin/env python3
"""runner: totals -- does the model keep its own numbers straight?

Long reports have a failure mode ordinary benchmarks never touch: every section
is correct, and the summary at the top contradicts them. It shows up in real
work (a review says "6 defects in the code" in section 5 and "0 chunk bugs" in
section 7) and it makes an otherwise useful report untrustworthy.

Protocol: the model receives a small, exact dataset and must write a report with
several sections plus a summary line. The dataset is constructed so that the
correct totals are unambiguous. Scoring checks the arithmetic, not the prose:

* ``per_section``  -- each section's stated count matches the data
* ``grand_total``  -- the summary line matches the sum of the sections
* ``self_consistent`` -- summary numbers agree with the numbers used in sections

A model that writes beautiful prose with wrong totals fails. A terse model with
correct numbers passes. That is the intent.

Usage:
  python -m runners.totals --url http://127.0.0.1:1234 --model my-model
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

from bench_common import Client, EngineError

CATEGORIES = ["logic errors", "leaks", "error handling gaps", "dead code"]

PROMPT = """You are writing a defect review from the data below.

DATA
{data}

Write the review with exactly four sections, one per category, and a closing
summary. Each section must state how many items it contains. The summary must
state the grand total and must agree with the sections.

Format the summary as a single line:
SUMMARY: total={total}, worst={category_with_most_items}

Do not invent items. Use the counts exactly as given.
""".replace("{total}", "{{total}}").replace("{category_with_most_items}", "{{category_with_most_items}}")


def make_data(rng: random.Random) -> dict:
    counts = {c: rng.randint(2, 9) for c in CATEGORIES}
    lines = []
    for cat, n in counts.items():
        lines.append(f"{cat} ({n} items):")
        for i in range(n):
            lines.append(f"  {i + 1}. {_item_text(cat, i, rng)}")
    worst = max(counts.items(), key=lambda kv: kv[1])[0]
    return {"counts": counts, "text": "\n".join(lines),
            "total": sum(counts.values()), "worst": worst}


def _item_text(cat: str, i: int, rng: random.Random) -> str:
    pool = {
        "logic errors": ["off-by-one in the page slice", "wrong default owner",
                         "comparison uses the wrong field", "branch never taken",
                         "loop bound reads the stale value", "timestamp compared as text",
                         "retry counter reset early", "guard inverted"],
        "leaks": ["file handle not closed", "connection kept in a module global",
                  "timer started and never stopped", "cache entry never evicted",
                  "thread pool not shut down", "listener not removed",
                  "buffer retained by closure", "temp dir not cleaned"],
        "error handling gaps": ["bare except swallows the cause", "no timeout on the call",
                                "error returned as a string", "exception ignored in a finally",
                                "retry without a limit", "no rollback after a partial write",
                                "log line without the trace", "status code not checked"],
        "dead code": ["unused import", "function no caller references",
                      "constant left from an old flag", "branch disabled by default",
                      "helper superseded by the inline copy", "test-only shim in production",
                      "commented block kept for reference", "unreachable return"],
    }
    options = pool[cat]
    return options[i % len(options)]


def score_totals(text: str, data: dict) -> dict:
    low = text.lower()

    # summary line
    m = re.search(r"SUMMARY:\s*total\s*=\s*(\d+)", text, re.I)
    claimed_total = int(m.group(1)) if m else None
    total_ok = claimed_total == data["total"]

    # worst category: match the phrase anywhere in the summary line or body
    worst_ok = data["worst"].lower() in low

    # per-section counts: the number may come before or after the category name
    # ("Logic Errors -- 8 items" or "8 items in logic errors"), so look on both
    # sides of the mention rather than assuming one order.
    section_ok = {}
    for cat, n in data["counts"].items():
        name = re.escape(cat.lower())
        # Allow a newline between the category header and its count: models put
        # the number on the next line ("### Leaks\nThis section has 6 items.").
        # [^.\n] instead of [^\n] would not help -- the header is followed by \n.
        after = re.search(name + r".{0,120}?(\d+)", low, re.DOTALL)
        before = re.search(r"(\d+)\s*(?:items|defects|issues|problems)?[^\n.]{0,40}?" + name, low)
        hit = None
        for candidate in (after, before):
            if candidate and int(candidate.group(1)) == n:
                hit = candidate
                break
        section_ok[cat] = hit is not None

    # self-consistency: the numbers the model used inside sections should sum to
    # the total it printed, not to some other number it invented on the way
    used = re.findall(r"(\d+)\s*(?:items|defects|issues|problems)", low)
    used_nums = [int(x) for x in used]
    sum_ok = claimed_total is not None and data["total"] in used_nums or total_ok

    return {
        "claimed_total": claimed_total,
        "expected_total": data["total"],
        "total_ok": total_ok,
        "worst_ok": worst_ok,
        "sections": section_ok,
        "sections_ok": sum(section_ok.values()),
        "sections_max": len(section_ok),
        "self_consistent": bool(total_ok and sum_ok),
        "correct": bool(total_ok and worst_ok and sum(section_ok.values()) >= 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1234")
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="unknown")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="results/totals.jsonl")
    args = ap.parse_args()

    client = Client(args.url, args.model, args.api_key)
    ok, msg = client.health()
    print(f"engine: {msg}")
    if not ok:
        raise SystemExit(2)

    data = make_data(random.Random(args.seed))
    print(f"dataset: {data['total']} items across {len(data['counts'])} categories\n")

    try:
        r = client.chat([{"role": "user", "content": PROMPT.format(data=data["text"])}],
                        max_tokens=1500, temperature=0.0)
    except EngineError as e:
        print(f"engine error: {e}")
        raise SystemExit(2) from None

    score = score_totals(r["text"], data)
    row = {"bench": "totals", "model": args.model, "quant": args.quant,
           "counts": data["counts"], "answer_head": r["text"] or "", **score}
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(row, fh, ensure_ascii=False, indent=2)

    print(f"expected total    {score['expected_total']}  claimed {score['claimed_total']} -> {score['total_ok']}")
    print(f"worst category    {data['worst']} -> {score['worst_ok']}")
    print(f"sections correct  {score['sections_ok']}/{score['sections_max']}")
    print(f"self-consistent   {score['self_consistent']}")
    print(f"CORRECT           {score['correct']}")
    print(f"->  {args.out}")


if __name__ == "__main__":
    main()
