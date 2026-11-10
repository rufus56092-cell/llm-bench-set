#!/usr/bin/env python3
"""runner: session -- work through a multi-file project without losing the thread.

The model is handed a brief and a directory of files, then asked to read them and
write a review. This is the shape of real work, and it fails differently from a
single prompt: models skip files, invent findings, lose earlier context, and
report a summary that contradicts the details they just wrote.

Three scores, all deterministic:

* ``files_read``   -- how many files the model actually opened (from the tool
                      trace). A model that answers without reading scores 0 here
                      no matter how good the prose is.
* ``bugs_found``   -- how many of the seeded defects appear in its findings.
                      Matched against ``re.search`` on the defect description's
                      key phrase, not on wording.
* ``consistency``  -- 1.0 if the model's own summary counts match the findings it
                      listed, 0.0 otherwise. A model can be right about every
                      file and still contradict itself in the wrap-up.

Usage:
  python -m runners.session --url http://127.0.0.1:1234 --model my-model
  python -m runners.session --url ... --model ... --data data/session
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from bench_common import Client, EngineError

# Phrases that signal the model spotted a given defect. Deliberately loose on
# wording, strict on substance: "no temp file" and "writes directly" both count,
# "the storage module handles files" does not.
BUG_SIGNALS = {
    "storage": ["without a temporary", "no temp", "truncat", "non-atomic", "not atomic"],
    "validation": ["empty dict", "empty object", "returns true for empty", "accepts empty"],
    "auth": ["constant-time", "constant time", "timing attack", "== instead of", "hashing compare"],
    "api": ["never caps", "no cap", "unbounded page", "page_size is not limited", "unbounded"],
    "cache": ["shrinks", "while iterating", "mutat.*iter", "dict changed size"],
    "scheduler": ["never resets", "not reset", "backoff grows", "delay keeps"],
    "metrics": ["not closed", "exception escapes", "timer leaks", "histogram drifts"],
    "export": ["never quotes", "no quoting", "comma in value", "unescaped"],
    "migrate": ["double-appl", "retried", "left registered", "re-applied"],
    "notify": ["timestamp", "not signed", "signature.*timestamp", "replay"],
    "search": ["lowercases the query", "not the stored", "case mismatch"],
    "retention": ["leaves.*files", "orphan.*attachment", "files remain"],
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the project directory",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "relative path"}},
            "required": ["path"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file into the project directory",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}]

SYSTEM = ("You are reviewing a Python service. Use read_file to open files before "
          "judging them -- do not guess at contents. When you have read the source "
          "modules, write your review to findings.md with write_file, then answer "
          "with a short summary that ends with 'FILES_READ: <n>' and "
          "'ISSUES_FOUND: <n>'.")


def read_file_tool(data_dir: Path, path: str) -> str:
    """Serve files from the data dir only. Path traversal is rejected."""
    target = (data_dir / path).resolve()
    try:
        target.relative_to(data_dir.resolve())
    except ValueError:
        return "error: path outside project"
    if not target.exists():
        return f"error: no such file {path}"
    return target.read_text(encoding="utf-8", errors="replace")[:8000]


def run_session(client: Client, data_dir: Path, max_turns: int = 40,
                max_tokens: int = 4096) -> dict:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    listing = "\n".join(f"- {f}" for f in meta["files"])

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Project directory listing:\n{listing}\n\n"
            f"Brief:\n{meta['brief']}\n\n"
            f"Read the source modules (chunk_*.py), the docs and the incident notes, "
            f"find real defects, and write findings.md. Keep the brief in mind -- I "
            f"will ask about it afterwards."},
    ]

    read_paths: list[str] = []
    written: dict[str, str] = {}
    turns = 0
    nudged = False

    for turns in range(1, max_turns + 1):
        r = client.chat(messages, max_tokens=max_tokens, temperature=0.0, tools=TOOLS)
        calls = r.get("tool_calls") or []
        if not calls:
            text = (r.get("text") or "").lower()
            # A model that says "now I will write the file" and stops has not
            # finished. One nudge is allowed: real agents loop until the work is
            # actually done, and a single pass without it scores models that
            # chatter as if they had produced output.
            if not written and not nudged:
                nudged = True
                messages.append({"role": "assistant", "content": r.get("content") or ""})
                messages.append({"role": "user", "content":
                                 "You have read the files but have not called write_file "
                                 "yet. Call write_file with path 'findings.md' now, then "
                                 "answer with the summary."})
                # The findings file is the artifact being scored, and a model that
                # spends its whole budget on the review text emits an empty
                # tool-call turn. Give the writing turn a larger budget.
                max_tokens = max(max_tokens, 6144)
                continue
            return {"messages": messages, "read_paths": read_paths,
                    "written": written, "final": r, "turns": turns,
                    "file_contents": {n: (data_dir / n).read_text(encoding="utf-8",
                                                                 errors="replace")
                                      for n in meta["files"] if (data_dir / n).is_file()}}

        messages.append({"role": "assistant", "content": r.get("content") or "",
                         "tool_calls": calls})
        for tc in calls:
            fn = tc.get("function") or tc
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "read_file":
                p = str(args.get("path", ""))
                content = read_file_tool(data_dir, p)
                if not content.startswith("error:"):
                    read_paths.append(p)
                result = content
            elif name == "write_file":
                p = str(args.get("path", ""))
                written[p] = str(args.get("content", ""))
                # Explicit "done" feedback: without it models keep re-writing the
                # same file, waiting for confirmation that never comes.
                result = (f"ok: wrote {len(written[p])} chars to {p}. "
                          f"File saved. If you have read all source modules and recorded "
                          f"your findings in findings.md, reply with your plain-text "
                          f"summary now (no tool calls).")
            else:
                result = f"error: unknown tool {name}"
            messages.append({"role": "tool", "tool_call_id": tc.get("id", "call_0"),
                             "content": result[:8000]})

    return {"messages": messages, "read_paths": read_paths, "written": written,
            "file_contents": {n: (data_dir / n).read_text(encoding="utf-8", errors="replace")
                              for n in meta["files"] if (data_dir / n).is_file()},
            "final": {"text": "", "finish_reason": "max_turns"}, "turns": turns}


def score_session(result: dict, meta: dict) -> dict:
    """Deterministic scoring of one session run."""
    brief_code = re.search(r"SESSION CODE:\s*([A-Z0-9\-]+)", meta["brief"])
    brief_code = brief_code.group(1) if brief_code else ""

    findings = ""
    for name, content in result["written"].items():
        if name.endswith(".md"):
            findings += "\n" + content
    if not findings:
        findings = result["final"].get("text", "")

    # ---- guard against degenerate input
    # If every source chunk has the same body, the seeded-defect score below
    # measures the generator, not the model: only the storage defect exists
    # twelve times over. Detect it and refuse to report a number.
    bodies = {}
    for name, content in result.get("file_contents", {}).items():
        body = re.sub(r'^""".*?"""', "", content, count=1, flags=re.DOTALL).strip()
        bodies.setdefault(body, []).append(name)
    biggest = max(bodies.values(), key=len) if bodies else []
    degenerate = len(biggest) > max(2, len(bodies and result.get("file_contents", {})) // 2)
    data_ok = not degenerate

    # ---- defects found (only scored when the corpus is not degenerate)
    found, missed = [], []
    low = findings.lower()
    for module, signals in BUG_SIGNALS.items():
        if any(re.search(sig, low) for sig in signals):
            found.append(module)
        else:
            missed.append(module)

    # ---- files actually opened
    read = {Path(p).name for p in result["read_paths"]}
    expected_files = {Path(f).name for f in meta["files"] if f.startswith("chunk_")}
    files_read = len(read & expected_files)
    files_expected = len(expected_files)

    # ---- self-consistency: does the summary agree with its own findings?
    final_text = result["final"].get("text", "") or ""
    m = re.search(r"ISSUES_FOUND:\s*(\d+)", final_text)
    claimed = int(m.group(1)) if m else None
    # Count findings however they are formatted: bullets, bold "F1 —" headers,
    # or numbered lines. The first version counted only "* " bullets and
    # reported a model that used "1." numbering as having listed nothing.
    listed = len(re.findall(r"^\s*(?:[-*•]|\d+[.)]|\*\*F\d+|\|?\s*F\d+\s*[—-])",
                            findings, re.MULTILINE))
    if listed == 0:
        listed = len(re.findall(r"\bF\d+\b", findings))
    consistent = claimed is not None and listed > 0 and abs(claimed - listed) <= max(2, listed // 5)

    # ---- did it remember the brief?
    remembered = bool(brief_code) and brief_code.lower() in (final_text + findings).lower()

    return {
        "data_ok": data_ok,
        "degenerate_corpus": degenerate,
        "duplicate_bodies": len(biggest),
        "files_read": files_read,
        "files_expected": files_expected,
        "bugs_found": len(found) if data_ok else None,
        "bugs_expected": len(BUG_SIGNALS),
        "found_modules": found if data_ok else [],
        "missed_modules": missed if data_ok else [],
        "consistency": bool(consistent),
        "claimed_issues": claimed,
        "listed_issues": listed,
        "remembered_brief": remembered,
        "turns": result["turns"],
        "read_paths": len(result["read_paths"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1234")
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="unknown")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY"))
    ap.add_argument("--data", default="data/session")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--out", default="results/session.jsonl")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not (data_dir / "meta.json").exists():
        raise SystemExit(f"no data at {data_dir}; run runners/make_session_data.py first")

    client = Client(args.url, args.model, args.api_key)
    ok, msg = client.health()
    print(f"engine: {msg}\n")
    if not ok:
        raise SystemExit(2)

    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    print(f"project: {len(meta['files'])} files, {len(BUG_SIGNALS)} seeded defects, "
          f"max {args.max_turns} turns\n")

    os.makedirs("results", exist_ok=True)
    try:
        result = run_session(client, data_dir, args.max_turns)
    except EngineError as e:
        print(f"engine error: {e}")
        raise SystemExit(2) from None

    row = {
        "bench": "session", "model": args.model, "quant": args.quant,
        **score_session(result, meta),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(row, fh, ensure_ascii=False, indent=2)

    print(f"files read      {row['files_read']}/{row['files_expected']}")
    print(f"defects found   {row['bugs_found']}/{row['bugs_expected']}  "
          f"{', '.join(row['found_modules']) or '-'}")
    print(f"missed          {', '.join(row['missed_modules']) or '-'}")
    print(f"self-consistent {row['consistency']}  "
          f"(claimed {row['claimed_issues']}, listed {row['listed_issues']})")
    print(f"brief recalled  {row['remembered_brief']}")
    print(f"turns           {row['turns']}")
    print(f"->  {args.out}")


if __name__ == "__main__":
    main()
