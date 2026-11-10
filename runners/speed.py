#!/usr/bin/env python3
"""runner: speed -- prefill, decode and time-to-first-token.

Measures three things separately, because they answer different questions:

* prefill  -- how long before the first token appears on a big prompt
* decode   -- how fast the model keeps talking once it started
* ttft     -- what the user actually feels when they hit enter

Server-reported timings are used when the backend exposes them (llama.cpp and
LM Studio report ``prompt_per_second`` / ``predicted_per_second``); otherwise
the numbers are derived from ``usage`` and wall clock and marked ``client``.

Three prompt sizes so the prefill curve is visible rather than a single point.

Usage:
  python -m runners.speed --url http://127.0.0.1:1234 --model my-model
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request

from bench_common import Client, EngineError

# short, medium, large -- enough to show the prefill curve on any machine
SIZES = [512, 8192, 32768]
GEN_TOKENS = 256


def build(n_tokens: int) -> str:
    """Neutral prose filler, ~8.9 chars/token on this text."""
    sentence = ("The archive holds records from the northern district, "
                "sorted by year and indexed by hand. ")
    return (sentence * max(1, int(n_tokens * 8.9 / len(sentence))))[
           : int(n_tokens * 8.9)]


def stream_ttft(client: Client, prompt: str, max_tokens: int) -> dict:
    """One streaming request; returns TTFT and observed decode speed."""
    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        client.url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=client.timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                chunks += 1
    total = time.time() - t0
    decode = (chunks / (total - ttft)) if (ttft and total > ttft and chunks > 1) else None
    return {"ttft_s": round(ttft, 3) if ttft else None,
            "stream_wall_s": round(total, 2),
            "stream_chunks": chunks,
            "stream_decode_tok_s": round(decode, 1) if decode else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1234")
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="unknown")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY"))
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--gen-tokens", type=int, default=GEN_TOKENS)
    ap.add_argument("--stream", action="store_true",
                    help="also measure TTFT over SSE (needs server stream support)")
    ap.add_argument("--out", default="results/speed.jsonl")
    args = ap.parse_args()

    client = Client(args.url, args.model, args.api_key)
    ok, msg = client.health()
    print(f"engine: {msg}\n")
    if not ok:
        raise SystemExit(2)

    sizes = [int(x) for x in args.sizes.split(",")]
    os.makedirs("results", exist_ok=True)
    rows: list[dict] = []

    with open(args.out, "w", encoding="utf-8") as fh:
        for n in sizes:
            prompt = build(n)
            try:
                r = client.chat([{"role": "user", "content": prompt}],
                                max_tokens=args.gen_tokens, temperature=0.0)
            except EngineError as e:
                print(f"{n:>7} tokens  ERROR {e}")
                continue
            row = {
                "bench": "speed", "model": args.model, "quant": args.quant,
                "target_prompt_tokens": n,
                "real_prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"),
                "prefill_tok_s": r.get("prefill_tok_s"),
                "decode_tok_s": r.get("decode_tok_s"),
                "speed_source": r.get("speed_source"),
                "wall_s": r.get("wall_s"),
            }
            if args.stream:
                try:
                    row.update(stream_ttft(client, prompt, args.gen_tokens))
                except Exception as e:                        # noqa: BLE001
                    row["stream_error"] = str(e)[:100]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            rows.append(row)
            print(f"{n:>7} tok target -> real {str(row['real_prompt_tokens']):>7} | "
                  f"prefill {str(row['prefill_tok_s']):>7} tok/s | "
                  f"decode {str(row['decode_tok_s']):>6} tok/s | "
                  f"wall {row['wall_s']:>7.1f}s", flush=True)

    if not rows:
        return
    dec = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
    pre = [r["prefill_tok_s"] for r in rows if r.get("prefill_tok_s")]
    print("\n" + "=" * 66)
    if dec:
        print(f"decode: mean {statistics.mean(dec):.1f}  median {statistics.median(dec):.1f} "
              f" max {max(dec):.1f} tok/s")
    if pre:
        print(f"prefill: mean {statistics.mean(pre):.0f}  min {min(pre):.0f} "
              f" max {max(pre):.0f} tok/s")
    ttf = [r["ttft_s"] for r in rows if r.get("ttft_s")]
    if ttf:
        print(f"ttft:   mean {statistics.mean(ttf):.2f}s  max {max(ttf):.2f}s")

    summary = {
        "bench": "speed", "model": args.model, "quant": args.quant,
        "decode_tok_s_mean": round(statistics.mean(dec), 1) if dec else None,
        "decode_tok_s_median": round(statistics.median(dec), 1) if dec else None,
        "prefill_tok_s_mean": round(statistics.mean(pre), 1) if pre else None,
        "ttft_s_mean": round(statistics.mean(ttf), 3) if ttf else None,
        "points": rows,
    }
    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"->  {args.out}")


if __name__ == "__main__":
    main()
