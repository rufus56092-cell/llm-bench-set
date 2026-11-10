#!/usr/bin/env python3
"""bench.py -- one entry point for every test in the set.

    python bench.py --url http://127.0.0.1:1234 --model my-model --quant IQ2_XS
    python bench.py --url ... --model ... --only quality,longctx
    python bench.py --compare results/*_summary.json

The result of every run is a flat summary JSON plus a detailed .jsonl, so any
two machines can be diffed without a database.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BENCHES = {
    "quality":   "runners.quality",
    "longctx":   "runners.longctx",
    "speed":     "runners.speed",
    "session":   "runners.session",
    "memory":    "runners.memory",
    "totals":    "runners.totals",
}

ORDER = ["speed", "quality", "longctx", "session", "memory", "totals"]


def run_one(name: str, args: argparse.Namespace) -> bool:
    mod = BENCHES[name]
    if not Path(mod.replace(".", "/") + ".py").exists():
        print(f"  [skip] {name}: runner not present")
        return False
    cmd = [sys.executable, "-m", mod,
           "--url", args.url, "--model", args.model, "--quant", args.quant,
           "--out", f"results/{name}.jsonl"]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    if name == "quality" and args.only_cases:
        cmd += ["--only", args.only_cases]
    if name == "longctx":
        cmd += ["--sizes", args.sizes, "--depths", args.depths]
    if name == "speed" and args.stream:
        cmd += ["--stream"]

    print(f"\n{'=' * 66}\n>>> {name}\n{'=' * 66}")
    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f"<<< {name}: {'ok' if rc == 0 else f'exit {rc}'} "
          f"in {(time.time() - t0) / 60:.1f} min")
    return rc == 0


def summarise(model: str, quant: str, ran: list[str]) -> None:
    """Merge per-bench summaries into one flat file."""
    out = {"model": model, "quant": quant, "benchmarks": {}}
    for name in ran:
        p = Path(f"results/{name}_summary.json")
        if p.exists():
            with open(p, encoding="utf-8") as f:
                out["benchmarks"][name] = json.load(f)
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dest = f"results/{model.replace('/', '_')}__{quant}_run.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\ncombined result -> {dest}")


def compare(paths: list[str]) -> None:
    """Print a side-by-side table of several run files."""
    runs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            runs.append((Path(p).stem, json.load(f)))

    def cell(run: dict, *keys, default="-"):
        cur = run
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    rows = [
        ("quality overall",   ("benchmarks", "quality", "overall")),
        ("quality scored",    ("benchmarks", "quality", "tasks_scored")),
        ("longctx recall",    ("benchmarks", "longctx", "recall")),
        ("largest ctx found", ("benchmarks", "longctx", "by_length")),
        ("decode tok/s",      ("benchmarks", "speed", "decode_tok_s_mean")),
        ("prefill tok/s",     ("benchmarks", "speed", "prefill_tok_s_mean")),
        ("ttft s",            ("benchmarks", "speed", "ttft_s_mean")),
        ("session files",     ("benchmarks", "session", "files_read")),
        ("memory kept",       ("benchmarks", "memory", "thread_kept")),
        ("totals correct",    ("benchmarks", "totals", "correct")),
    ]

    width = max(len(r[0]) for r in rows) + 2
    head = f"{'metric':<{width}}" + "".join(f"{n[:18]:>20}" for n, _ in runs)
    print(head)
    print("-" * len(head))
    for label, keys in rows:
        line = f"{label:<{width}}"
        for _, run in runs:
            v = cell(run, *keys)
            if isinstance(v, dict):
                v = len(v)
            line += f"{str(v):>20}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:1234")
    ap.add_argument("--model")
    ap.add_argument("--quant", default="unknown")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY"))
    ap.add_argument("--only", default="", help="comma separated subset of benches")
    ap.add_argument("--only-cases", default="", help="comma separated quality case ids")
    ap.add_argument("--sizes", default="8000,32000,64000,100000")
    ap.add_argument("--depths", default="0.05,0.5,0.95")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--compare", nargs="+", help="compare finished run files")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.compare:
        compare(args.compare)
        return

    if not args.model:
        ap.error("--model is required unless you pass --compare")

    chosen = ([x.strip() for x in args.only.split(",") if x.strip()]
              if args.only else ORDER)

    print(f"model: {args.model}   quant: {args.quant}")
    print(f"target: {args.url}")
    print(f"benches: {', '.join(chosen)}")

    ran = [n for n in chosen if run_one(n, args)]
    if ran:
        summarise(args.model, args.quant, ran)
    print(f"\ndone: {len(ran)}/{len(chosen)} benchmarks")


if __name__ == "__main__":
    main()
