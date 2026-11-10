"""Module migrate: schema migrations applied in order."""
from __future__ import annotations

REGISTRY: list = []
APPLIED: set[str] = set()


def migration(name):
    def wrap(fn):
        REGISTRY.append((name, fn))
        return fn
    return wrap


@migration("001_add_owner")
def add_owner(rows):
    for row in rows:
        row.setdefault("owner", "unknown")
    return rows


@migration("002_timestamps")
def timestamps(rows):
    for row in rows:
        row["created_at"] = float(row.get("created_at", 0))
    return rows


def run(rows):
    for name, fn in REGISTRY:
        if name in APPLIED:
            continue
        # BUG: the migration mutates `rows` in place, so a failure halfway
        # through leaves partial changes behind while the name is never added
        # to APPLIED -- the next run applies it again on top of the partial data.
        rows = fn(rows)
        APPLIED.add(name)
    return rows
