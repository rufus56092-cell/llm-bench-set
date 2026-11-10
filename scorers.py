#!/usr/bin/env python3
"""Scorers: turn a model answer into a 0..5 score.

Every scorer is deterministic and offline except ``judge`` -- and even that one
falls back to keyword/constraint checks when the case carries machine-checkable
fields, so a benchmark run never needs a paid API.

Scale is fixed at 0..5 so categories can be compared and averaged:
  5.0 correct / all constraints met
  3.0 partly correct (expected substring present, some constraints missed)
  1.0 ran but wrong (unit test executed and failed)
  0.0 no answer / crashed
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

MAX_TOKENS_MSG = 5.0


# --------------------------------------------------------------------- helpers
def _json_equal(a: str, b: str) -> bool:
    try:
        return json.loads(a) == json.loads(b)
    except Exception:
        return False


_DAYS = {"monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu",
         "friday": "fri", "saturday": "sat", "sunday": "sun"}


def _norm_days(s: str) -> str:
    out = s.lower()
    for full, short in _DAYS.items():
        out = out.replace(full, short)
    return out


def score_exact(response: str, expected: str) -> tuple[float, str]:
    clean = (response or "").strip().strip("`").strip()
    exp = (expected or "").strip()
    if _json_equal(clean, exp):
        return 5.0, "json-equal"
    if _norm_days(clean) == _norm_days(exp):
        return 5.0, "day-normalised-equal"
    if clean.lower() == exp.lower():
        return 5.0, "case-insensitive-equal"
    if clean == exp:
        return 5.0, "exact"

    # A short expected answer buried in prose ("...= 1/2.") is correct, but an
    # unanchored substring test also fires inside unrelated numbers: '1/2' would
    # match '1/20'. Guard short numeric/fraction answers with digit boundaries.
    if _short_answer(exp):
        if _contains_answer(clean, exp):
            return 3.0, "expected-in-prose"
        return 0.0, "no-match"

    if exp.lower() in clean.lower():
        return 3.0, "expected-substring"
    norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
    if norm(exp) in norm(clean):
        return 3.0, "expected-substring-normalised"
    return 0.0, "no-match"


_SHORT_RE = re.compile(r"^[\d\s/.,%+\-]+$")


def _short_answer(exp: str) -> bool:
    """Numeric or fraction answers need boundary-aware matching."""
    return len(exp) <= 12 and bool(_SHORT_RE.match(exp))


def _contains_answer(text: str, exp: str) -> bool:
    """Match exp as a standalone number/fraction, allowing surrounding prose.

    List answers ('5,12') are split on commas: a model that writes '5, 12'
    with a space is not wrong, and a plain boundary regex on the whole string
    would miss it.
    """
    parts = [p.strip() for p in exp.split(",") if p.strip()] if "," in exp else [exp.strip()]
    if len(parts) > 1:
        return all(re.search(rf"(?<![\d.]){re.escape(p)}(?![\d.])", text) for p in parts)
    needle = re.escape(exp.strip())
    # Trailing dot is ambiguous: it may be a decimal ('0.25') or a sentence
    # period ('= 1/2.'). Reject only when the dot starts another number.
    pattern = rf"(?<![\d/.])({needle})(?![\d/]|\.\d)"
    return re.search(pattern, text) is not None


def score_keywords(response: str, keywords: list[str]) -> tuple[float, str]:
    if not keywords:
        return 0.0, "no-keywords"
    low = (response or "").lower()
    found = sum(1 for kw in keywords if kw.lower() in low)
    return round(5.0 * found / len(keywords), 2), f"{found}/{len(keywords)} keywords"


def score_unit_tests(response: str, test_code: str, timeout: int = 15) -> tuple[float, str]:
    """Run the model's code against the case's test code.

    A failure score is 1.0, not 0.0: the model *did* produce runnable Python,
    which is a materially different outcome from an empty answer.
    """
    code = response or ""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", code, re.DOTALL)
    if m:
        code = m.group(1)
    else:
        # No fence: pull out the definition if there is one. Require an actual
        # def/class -- an earlier version kept every line after the first
        # "import", which turned ordinary prose into a "tested" snippet and
        # scored a refusal as runnable code.
        lines = code.strip().split("\n")
        start = next((i for i, l in enumerate(lines)
                      if l.lstrip().startswith(("def ", "class ", "async def "))), None)
        if start is None:
            return 0.0, "no-code"
        code = "\n".join(lines[start:])

    if not code.strip():
        return 0.0, "no-code"

    try:
        r = subprocess.run([sys.executable, "-c", code + "\n\n" + test_code],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 0.0, "unit-test-timeout"
    except Exception as e:                                    # noqa: BLE001
        return 0.0, f"unit-test-error: {type(e).__name__}"

    if r.returncode == 0:
        return 5.0, "tests-passed"
    return 1.0, "tests-failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:80]


def score_constraints(response: str, constraints: dict) -> tuple[float, str]:
    text = response or ""
    score, notes = 5.0, []

    def penalise(amount: float, why: str) -> None:
        nonlocal score
        score -= amount
        notes.append(why)

    if "word_count" in constraints:
        w = len(text.split())
        if w != constraints["word_count"]:
            penalise(2, f"word_count {w}!={constraints['word_count']}")

    if "word_range" in constraints:
        lo, hi = constraints["word_range"]
        w = len(text.split())
        if not (lo <= w <= hi):
            penalise(2, f"word_range {w} not in [{lo},{hi}]")

    if "line_count" in constraints:
        n = len([l for l in text.strip().split("\n") if l.strip()])
        if n != constraints["line_count"]:
            penalise(2, f"line_count {n}!={constraints['line_count']}")

    if "bullet_count" in constraints:
        b = [l for l in text.strip().split("\n") if l.strip().startswith(("-", "*", "•"))]
        if len(b) != constraints["bullet_count"]:
            penalise(1, f"bullet_count {len(b)}!={constraints['bullet_count']}")

    if "words_per_bullet" in constraints:
        for b in [l for l in text.strip().split("\n") if l.strip().startswith(("-", "*", "•"))]:
            w = len(b.strip().lstrip("-*• ").split())
            if w != constraints["words_per_bullet"]:
                penalise(0.5, f"words_per_bullet {w}!={constraints['words_per_bullet']}")

    low = text.lower()
    for key, label in (("required_words", "missing word"),
                       ("required_elements", "missing element")):
        for w in constraints.get(key, []):
            if w.lower() not in low:
                penalise(1, f"{label} '{w}'")

    for key, label in (("forbidden_words", "forbidden word"),
                       ("forbidden_elements", "forbidden element")):
        for w in constraints.get(key, []):
            if w.lower() in low:
                penalise(1, f"{label} '{w}'")

    for ch in constraints.get("forbidden_chars", []):
        if ch in text:
            penalise(1, f"forbidden char '{ch}'")

    return max(0.0, score), ("; ".join(notes) if notes else "all constraints met")


def _arg_match(actual, want) -> bool:
    """`_contain` means contains, not equals: '15:00' should match a full ISO stamp."""
    if actual is None:
        return False
    a, w = str(actual).strip(), str(want).strip()
    if a == w:
        return True
    try:
        return float(a) == float(w)
    except (TypeError, ValueError):
        return w.lower() in a.lower()


def score_tool_trace(result: dict, case: dict) -> tuple[float, str]:
    trace = result.get("tool_calls_trace") or []
    calls = [tc for turn in trace for tc in (turn or [])]
    names = [tc.get("function_name") for tc in calls]
    expected = case.get("expected_tool_call")
    score, note = 0.0, ""

    if expected:
        if expected not in names:
            return 0.0, f"expected tool '{expected}' not called (got {names})"
        score += 2.5
        want_args = case.get("expected_args_contain") or {}
        if want_args:
            for tc in calls:
                if tc.get("function_name") == expected:
                    try:
                        got = json.loads(tc.get("function_args") or "{}")
                    except json.JSONDecodeError:
                        got = {}
                    hits = sum(1 for k, v in want_args.items() if _arg_match(got.get(k), v))
                    score += 2.5 * hits / len(want_args)
                    note = f"args {hits}/{len(want_args)}"
                    break

    kw = case.get("expected_keywords") or []
    if kw:
        ks, _ = score_keywords(result.get("text", ""), kw)
        score = (score + ks) / 2 if score > 0 else ks

    return min(score, MAX_TOKENS_MSG), note or "tool trace ok"


def is_invalid(result: dict, case: dict) -> tuple[bool, str]:
    if result.get("finish_reason") == "error":
        return True, "engine error"
    if not (result.get("text") or "").strip():
        return True, "empty answer"
    if result.get("finish_reason") == "length" and case.get("category") != "creative":
        # Some cases legitimately need a long chain of thought before the answer.
        # Cutting them at a fixed cap measures the cap, not the model, so those
        # are marked unscored rather than wrong.
        if case.get("long_expected"):
            return True, "truncated at cap (long_expected case, needs more max_tokens)"
        return True, "truncated by max_tokens"
    return False, ""


def score_case(result: dict, case: dict) -> tuple[float | None, str, str]:
    """Returns (score, scoring_method, note). score None => needs a judge."""
    bad, why = is_invalid(result, case)
    if bad:
        return None, "invalid", why

    text = result.get("text", "")
    scoring = case.get("scoring", "judge")

    if scoring == "exact" and "expected" in case:
        s, n = score_exact(text, case["expected"])
        return s, "exact", n
    if scoring == "judge_keyword" and case.get("expected_keywords"):
        s, n = score_keywords(text, case["expected_keywords"])
        return s, "keyword", n
    if scoring == "unit_tests" and case.get("test_code"):
        s, n = score_unit_tests(text, case["test_code"])
        return s, "unit_test", n
    if scoring in ("constraint_check", "judge_constraint") and case.get("constraints"):
        s, n = score_constraints(text, case["constraints"])
        return s, "constraint", n
    if scoring in ("tool_trace_exact", "tool_trace_judge"):
        s, n = score_tool_trace(result, case)
        return s, "tool_trace", n
    if scoring == "judge":
        if case.get("constraints"):
            s, n = score_constraints(text, case["constraints"])
            return s, "constraint", n
        if case.get("expected_keywords"):
            s, n = score_keywords(text, case["expected_keywords"])
            return s, "keyword", n

    return None, "needs_judge", "no machine-checkable criterion"
