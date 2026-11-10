#!/usr/bin/env python3
"""runner: quality -- general capability across 78 tasks in 6 categories.

Each case is a prompt plus a machine-checkable criterion. Three repeats per
case with a fixed seed list: temperature 0 makes repeats mostly identical for
deterministic categories and explores variety for creative ones.

Cases whose only criterion is a human "judge" are reported separately and are
NOT folded into the score -- that keeps the number comparable between machines
without requiring a paid judge model.

Usage:
  python -m runners.quality --url http://127.0.0.1:1234 --model my-model
  python -m runners.quality --url ... --model ... --only C1,C2,M1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from bench_common import Client, EngineError
from scorers import score_case

REPEATS = 3
DETERMINISTIC = [0.0, 0.0, 0.0]
CREATIVE = [0.7, 0.8, 0.9]
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "2048"))


def load_cases(path: str, only: set[str] | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if only:
        cases = [c for c in cases if c["id"] in only]
    return cases


def run_tool_case(client: Client, case: dict) -> dict:
    """Multi-turn tool loop: let the model call tools until it answers.

    Capped at 4 turns so a model that loops forever fails fast instead of
    burning an hour of GPU time. When the last turn is still a tool call, the
    tool name is copied into ``text`` so the answer is not scored as empty --
    the trace is the answer in that case.
    """
    messages = [{"role": "user", "content": case["prompt"]}]
    trace: list[list[dict]] = []
    last: dict = {}
    for _ in range(4):
        last = client.chat(messages, max_tokens=MAX_TOKENS, temperature=0.0,
                           tools=case.get("tools"))
        calls = last.get("tool_calls") or []
        if not calls:
            break
        norm = []
        for tc in calls:
            fn = tc.get("function") or tc
            args = fn.get("arguments")
            norm.append({
                "function_name": fn.get("name"),
                "function_args": args if isinstance(args, str) else json.dumps(args or {}),
            })
        trace.append(norm)
        # feed a stub result so the model can continue; we grade the trace, not the tool
        messages.append({"role": "assistant", "content": last.get("content") or "",
                         "tool_calls": calls})
        for tc in calls:
            fn = tc.get("function") or tc
            messages.append({"role": "tool", "tool_call_id": tc.get("id", "call_0"),
                             "content": json.dumps({"ok": True, "stub": True})})

    last["tool_calls_trace"] = trace
    if not (last.get("text") or "").strip() and trace:
        # Model stopped on a tool call. That is a valid outcome for "call this
        # tool" cases, so record the called names as the observable answer.
        called = ", ".join(tc["function_name"] for turn in trace for tc in turn)
        last["text"] = f"[tool_calls: {called}]"
        last["finish_reason"] = "tool_calls"
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1234")
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="unknown", help="label only, recorded in the result")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY"))
    ap.add_argument("--cases", default="cases/quality.json")
    ap.add_argument("--only", default="", help="comma separated case ids")
    ap.add_argument("--out", default="results/quality.jsonl")
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    client = Client(args.url, args.model, args.api_key, args.timeout)
    ok, msg = client.health()
    print(f"engine: {msg}")
    if not ok:
        raise SystemExit(2)

    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    cases = load_cases(args.cases, only)
    print(f"cases: {len(cases)} x {REPEATS} repeats, max_tokens={MAX_TOKENS}\n")

    os.makedirs("results", exist_ok=True)
    rows: list[dict] = []
    t_start = time.time()

    with open(args.out, "w", encoding="utf-8") as fh:
        for i, case in enumerate(cases, 1):
            cat = case.get("category", "?")
            temps = CREATIVE if cat == "creative" else DETERMINISTIC
            for rep, temp in enumerate(temps[:REPEATS]):
                t0 = time.time()
                try:
                    if case.get("tools"):
                        r = run_tool_case(client, case)
                    else:
                        r = client.chat(
                            [{"role": "user", "content": case["prompt"]}],
                            max_tokens=MAX_TOKENS, temperature=temp)
                except EngineError as e:
                    r = {"finish_reason": "error", "text": "", "error": str(e)}

                score, method, note = score_case(r, case)
                row = {
                    "bench": "quality",
                    "model": args.model,
                    "quant": args.quant,
                    "id": case["id"],
                    "category": cat,
                    "repeat": rep,
                    "temperature": temp,
                    "score": score,
                    "method": method,
                    "note": note,
                    "finish_reason": r.get("finish_reason"),
                    "prompt_tokens": r.get("prompt_tokens"),
                    "completion_tokens": r.get("completion_tokens"),
                    "decode_tok_s": r.get("decode_tok_s"),
                    "prefill_tok_s": r.get("prefill_tok_s"),
                    "speed_source": r.get("speed_source"),
                    "wall_s": r.get("wall_s") or round(time.time() - t0, 2),
                    "answer_head": (r.get("text") or "")[:300],
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(row)

                mark = f"{score:>4.1f}" if score is not None else " n/a"
                print(f"[{i:>2}/{len(cases)}] {case['id']:<3} {cat:<11} "
                      f"{mark}  {r.get('decode_tok_s') or '-':>6} tok/s  "
                      f"{row['wall_s']:>7.1f}s  {note[:48]}", flush=True)

    # ---------------------------------------------------------------- summary
    scored = [r for r in rows if r["score"] is not None]
    print("\n" + "=" * 66)
    if not scored:
        print("no scorable results")
        return

    by_cat: dict[str, list[float]] = {}
    for r in scored:
        by_cat.setdefault(r["category"], []).append(r["score"])

    print(f"{'category':<14}{'score':>8}{'tasks':>8}")
    for cat, vals in sorted(by_cat.items(), key=lambda kv: -statistics.mean(kv[1])):
        print(f"{cat:<14}{statistics.mean(vals):>8.2f}{len(vals):>8}")

    overall = statistics.mean(r["score"] for r in scored)
    speeds = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
    print("-" * 66)
    print(f"{'OVERALL':<14}{overall:>8.2f}{len(scored):>8}")
    if speeds:
        print(f"decode tok/s: mean {statistics.mean(speeds):.1f}  "
              f"median {statistics.median(speeds):.1f}  max {max(speeds):.1f}")
    print(f"elapsed: {(time.time() - t_start) / 60:.1f} min  ->  {args.out}")

    summary = {
        "bench": "quality", "model": args.model, "quant": args.quant,
        "overall": round(overall, 3),
        "by_category": {k: round(statistics.mean(v), 3) for k, v in by_cat.items()},
        "tasks_scored": len(scored),
        "tasks_unscored": len(rows) - len(scored),
        "decode_tok_s_mean": round(statistics.mean(speeds), 1) if speeds else None,
        "elapsed_min": round((time.time() - t_start) / 60, 1),
    }
    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
