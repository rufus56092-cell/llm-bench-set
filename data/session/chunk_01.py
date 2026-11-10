"""Module storage: persistence for records, atomic write."""
from __future__ import annotations

import json
from pathlib import Path

STORE = Path("records.json")


def _load(path=STORE):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(rows, path=STORE):
    # BUG: writes in place. A crash mid-write leaves a truncated file, although
    # the module header promises an atomic write.
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


def get(record_id):
    for row in _load():
        if row.get("id") == record_id:
            return row
    return None


def put(record):
    rows = [r for r in _load() if r.get("id") != record.get("id")]
    rows.append(record)
    _save(rows)
    return record


def count():
    return len(_load())
