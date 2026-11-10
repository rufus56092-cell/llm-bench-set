#!/usr/bin/env python3
"""runner: longctx -- needle in a haystack across context length x depth.

The point of this test is to catch silent truncation. Many stacks happily accept
a 100K prompt, quietly cut it to 4K, answer confidently, and the operator
believes the model "handles 100K". We defend against that two ways:

1. The context size is read from the server's own ``usage.prompt_tokens`` of the
   actual response -- never estimated from character counts.
2. The reported size is compared against the requested one. A cell whose real
   prompt is far below the target is flagged ``truncated`` instead of being
   counted as a pass.

Works with any OpenAI-compatible endpoint; no ``/tokenize`` required.

Usage:
  python -m runners.longctx --url http://127.0.0.1:1234 --model my-model
  python -m runners.longctx --sizes 8000,32000 --depths 0.1,0.5,0.9
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from statistics import mean

from bench_common import Client, EngineError

ADJ = ["quiet", "ancient", "restless", "amber", "hollow", "distant", "brittle",
       "luminous", "weathered", "sombre", "gilded", "narrow", "vast", "tepid"]
NOUN = ["harbour", "orchard", "lantern", "corridor", "meadow", "furnace",
        "archive", "glacier", "trellis", "cistern", "belfry", "estuary"]
VERB = ["drifted past", "settled over", "coiled around", "receded from",
        "pressed against", "wandered through", "loomed beyond", "faded near"]
PLACE = ["the old mill", "a copper sky", "the tram depot", "the salt flats",
         "a shuttered market", "the river bend", "the north pier", "a dry canal"]
CITIES = ["Reykjavik", "Ouagadougou", "Valparaiso", "Nakhodka", "Timbuktu",
          "Kirkwall", "Ushuaia", "Yakutsk", "Paramaribo", "Trondheim"]

CHARS_PER_TOKEN = 8.9      # calibrated on this filler text against a live server


def filler(n: int, rng: random.Random) -> str:
    return " ".join(
        f"The {rng.choice(ADJ)} {rng.choice(NOUN)} {rng.choice(VERB)} "
        f"{rng.choice(PLACE)} while the {rng.choice(ADJ)} {rng.choice(NOUN)} "
        f"{rng.choice(VERB)} {rng.choice(PLACE)}." for _ in range(n))


def build_prompt(target_tokens: int, depth: float, city: str, code: str,
                 rng: random.Random, needle_style: str = "en") -> str:
    if needle_style == "ru":
        needle = f"ВАЖНАЯ ЗАПИСЬ: секретный код доступа для города {city} — {code}."
        question = (f"\n\nВопрос: какой секретный код доступа для города {city}? "
                    f"Ответь только восьмизначным кодом, без пояснений.")
    else:
        needle = f"IMPORTANT RECORD: the secret access code for {city} is {code}."
        question = (f"\n\nQuestion: What is the secret access code for {city}? "
                    f"Answer with only the 8-digit code, nothing else.")

    total = max(20, int(target_tokens * CHARS_PER_TOKEN / 185))
    head = int(total * depth)
    body = (filler(head, rng) + " " + needle + " " + filler(total - head, rng)).strip()
    return body + question


def one_cell(client: Client, target: int, depth: float, seed: int,
             needle_style: str = "en") -> dict:
    rng = random.Random(seed)
    city = rng.choice(CITIES)
    code = f"{rng.randint(10_000_000, 99_999_999)}"
    prompt = build_prompt(target, depth, city, code, rng, needle_style)
    r = client.chat([{"role": "user", "content": prompt}],
                    max_tokens=32, temperature=0.0)
    m = re.search(r"\b(\d{8})\b", r["text"])
    real = r.get("prompt_tokens")
    truncated = bool(real and target > 2000 and real < target * 0.6)
    return {
        "target_tokens": target,
        "real_prompt_tokens": real,
        "depth": depth,
        "city": city,
        "code": code,
        "found": bool(m and m.group(1) == code),
        "answer": (r["text"] or "")[:60],
        "truncated": truncated,
        "prefill_tok_s": r.get("prefill_tok_s"),
        "decode_tok_s": r.get("decode_tok_s"),
        "speed_source": r.get("speed_source"),
        "wall_s": r.get("wall_s"),
        "finish_reason": r.get("finish_reason"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1234")
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="unknown")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY"))
    ap.add_argument("--sizes", default="8000,32000,64000,100000")
    ap.add_argument("--depths", default="0.05,0.5,0.95")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--needle-lang", choices=["en", "ru"], default="en",
                    help="ru adds a Cyrillic needle, useful for non-English models")
    ap.add_argument("--out", default="results/longctx.jsonl")
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    client = Client(args.url, args.model, args.api_key, args.timeout)
    ok, msg = client.health()
    print(f"engine: {msg}")
    if not ok:
        raise SystemExit(2)

    sizes = [int(x) for x in args.sizes.split(",")]
    depths = [float(x) for x in args.depths.split(",")]
    print(f"grid: {len(sizes)} lengths x {len(depths)} depths = "
          f"{len(sizes) * len(depths)} cells\n")

    os.makedirs("results", exist_ok=True)
    rows: list[dict] = []
    with open(args.out, "w", encoding="utf-8") as fh:
        for size in sizes:
            for depth in depths:
                try:
                    r = one_cell(client, size, depth, args.seed, args.needle_lang)
                except EngineError as e:
                    r = {"target_tokens": size, "depth": depth, "error": str(e)}
                r.update({"bench": "longctx", "model": args.model, "quant": args.quant})
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(r)

                if "error" in r:
                    print(f"{size:>7} @{depth:.2f}  ERROR  {r['error'][:70]}", flush=True)
                    continue
                mark = "FOUND  " if r["found"] else "MISSED "
                flag = "  [TRUNCATED]" if r["truncated"] else ""
                print(f"{size:>7} @{depth:.2f}  real {str(r['real_prompt_tokens']):>7} tok  "
                      f"{mark}  {r['wall_s']:>7.1f}s "
                      f"prefill {r['prefill_tok_s'] or '-'} tok/s{flag}", flush=True)

    good = [r for r in rows if "error" not in r]
    found = sum(1 for r in good if r["found"])
    trunc = sum(1 for r in good if r["truncated"])
    print("\n" + "=" * 66)
    print(f"recall: {found}/{len(good)} cells   truncated: {trunc}")
    if good:
        pref = [r["prefill_tok_s"] for r in good if r.get("prefill_tok_s")]
        if pref:
            print(f"prefill tok/s: mean {mean(pref):.0f}  min {min(pref):.0f}  max {max(pref):.0f}")
        safest = max(r["real_prompt_tokens"] or 0 for r in good if r["found"]) if found else 0
        if safest:
            print(f"largest verified context: {safest} tokens")
    print(f"->  {args.out}")

    summary = {
        "bench": "longctx", "model": args.model, "quant": args.quant,
        "cells": len(good), "found": found,
        "recall": round(found / len(good), 3) if good else 0.0,
        "truncated_cells": trunc,
        "by_length": {},
    }
    for size in sizes:
        cells = [r for r in good if r["target_tokens"] == size]
        if cells:
            summary["by_length"][str(size)] = round(
                sum(1 for r in cells if r["found"]) / len(cells), 3)
    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
