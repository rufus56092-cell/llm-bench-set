#!/usr/bin/env python3
"""Sanity tests for the scorers. Run: python tests_scorers.py

These exist because a benchmark that scores wrongly is worse than no benchmark:
it produces confident, comparable-looking numbers that are simply false. Every
guard here was added after a real mis-score on a live run.
"""
from __future__ import annotations

import sys

from scorers import (score_case, score_constraints, score_exact, score_keywords,
                     score_unit_tests)

FAILED: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {name}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok:
        FAILED.append(name)


# ------------------------------------------------------- exact / numeric guard
check("exact match", score_exact("1/2", "1/2")[0], 5.0)
check("answer in prose", score_exact("The result is 1/2.", "1/2")[0], 3.0)
check("1/2 must not match 1/20", score_exact("1/20", "1/2")[0], 0.0)
check("42 must not match 142", score_exact("142", "42")[0], 0.0)
check("42 in prose", score_exact("answer: 42", "42")[0], 3.0)
check("decimal exact", score_exact("0.25", "0.25")[0], 5.0)
check("0.25 must not match 0.253", score_exact("0.253", "0.25")[0], 0.0)
check("case insensitive", score_exact("Paris", "paris")[0], 5.0)
check("json structural", score_exact('{"a": 1, "b": 2}', '{"b":2,"a":1}')[0], 5.0)
check("day name normalised", score_exact("Monday", "Mon")[0], 5.0)
check("wrong answer", score_exact("3/4", "1/2")[0], 0.0)

# --------------------------------------------------------------------- keywords
check("all keywords", score_keywords("alpha beta", ["alpha", "beta"])[0], 5.0)
check("half keywords", score_keywords("alpha only", ["alpha", "beta"])[0], 2.5)
check("no keywords case", score_keywords("anything", [])[0], 0.0)

# ------------------------------------------------------------- unit test scoring
OK_CODE = "def add(a, b):\n    return a + b\n"
BAD_CODE = "def add(a, b):\n    return a - b\n"
TEST = "assert add(2, 3) == 5"
check("unit tests pass", score_unit_tests(OK_CODE, TEST)[0], 5.0)
check("unit tests run but fail", score_unit_tests(BAD_CODE, TEST)[0], 1.0)
check("no code at all", score_unit_tests("I cannot help with that.", TEST)[0], 0.0)
check("code in fence",
      score_unit_tests(f"```python\n{OK_CODE}```", TEST)[0], 5.0)

# ------------------------------------------------------------------ constraints
check("word count exact",
      score_constraints("one two three", {"word_count": 3})[0], 5.0)
check("word count off",
      score_constraints("one two", {"word_count": 3})[0], 3.0)
check("required word missing",
      score_constraints("nothing here", {"required_words": ["penguin"]})[0], 4.0)
check("forbidden word present",
      score_constraints("has penguin", {"forbidden_words": ["penguin"]})[0], 4.0)
check("bullet count", score_constraints("- a\n- b", {"bullet_count": 2})[0], 5.0)
check("constraints never negative",
      score_constraints("x", {"word_count": 99, "required_words": list("abcdefg")})[0], 0.0)

# ------------------------------------------------------------ invalid handling
case = {"id": "X", "category": "math", "scoring": "exact", "expected": "1"}
check("empty answer is invalid", score_case({"text": "", "finish_reason": "stop"}, case)[0], None)
check("engine error is invalid",
      score_case({"text": "1", "finish_reason": "error"}, case)[0], None)
check("length truncation is invalid",
      score_case({"text": "1", "finish_reason": "length"}, case)[0], None)
check("creative may be truncated",
      score_case({"text": "story", "finish_reason": "length"},
                 {"id": "W", "category": "creative", "scoring": "judge_keyword",
                  "expected_keywords": ["story"]})[0], 5.0)

print()
if FAILED:
    print(f"{len(FAILED)} failed: {', '.join(FAILED)}")
    sys.exit(1)
print("all scorer tests passed")
