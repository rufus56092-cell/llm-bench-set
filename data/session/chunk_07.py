"""Module metrics: counters and timers."""
from __future__ import annotations

import time

COUNTERS: dict[str, int] = {}
HISTOGRAMS: dict[str, list[float]] = {}


def incr(name, amount=1):
    COUNTERS[name] = COUNTERS.get(name, 0) + amount


def observe(name, value):
    HISTOGRAMS.setdefault(name, []).append(float(value))


class timer:
    def __init__(self, name):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        # BUG: records the elapsed time even when an exception escaped, so
        # failed calls pollute the latency histogram.
        observe(self.name, time.time() - self.start)
        return False


def snapshot():
    return {
        "counters": dict(COUNTERS),
        "histograms": {k: {"count": len(v), "sum": sum(v)} for k, v in HISTOGRAMS.items()},
    }
