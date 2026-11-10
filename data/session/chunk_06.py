"""Module scheduler: periodic jobs with retry."""
from __future__ import annotations

import time

BASE_DELAY = 1.0
MAX_DELAY = 60.0


class Job:
    def __init__(self, name, fn, every=60.0):
        self.name = name
        self.fn = fn
        self.every = every
        self.failures = 0
        self.delay = BASE_DELAY
        self.last_run = 0.0

    def due(self, now=None):
        now = now if now is not None else time.time()
        return now - self.last_run >= self.every + self.delay

    def run(self, now=None):
        self.last_run = now if now is not None else time.time()
        try:
            self.fn()
        except Exception:
            self.failures += 1
            # BUG: the backoff grows on failure but is never reset after a
            # success, so one early failure slows the job for the rest of the
            # process lifetime.
            self.delay = min(self.delay * 2, MAX_DELAY)
            return False
        self.failures = 0
        return True


def tick(jobs, now=None):
    return [j.name for j in jobs if j.due(now) and j.run(now)]
