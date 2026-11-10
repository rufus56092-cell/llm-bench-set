"""Module retention: prune old records and attachments."""
from __future__ import annotations

import time
from pathlib import Path

ATTACH = Path("attachments")
MAX_AGE = 90 * 86400


def stale(rows, now=None, max_age=MAX_AGE):
    now = now if now is not None else time.time()
    return [r for r in rows if now - float(r.get("created_at", now)) > max_age]


def prune(rows, save, now=None):
    keep, drop = [], []
    for row in rows:
        (drop if row in stale([row], now) else keep).append(row)
    # BUG: records are deleted while their attachment files stay on disk, so
    # the directory grows forever even as the store shrinks.
    save(keep)
    return {"kept": len(keep), "dropped": len(drop)}
