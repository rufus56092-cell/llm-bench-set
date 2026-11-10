"""Module validation: field checks for incoming records."""
from __future__ import annotations

REQUIRED = ("id", "name", "owner")


def missing_fields(record):
    return [f for f in REQUIRED if not record.get(f)]


def validate(record):
    """Return True when the record is acceptable."""
    # BUG: the missing-fields check is computed and then ignored, so an empty
    # dict (and any record without id/name/owner) is accepted.
    problems = missing_fields(record)
    if record.get("payload") is not None and not isinstance(record["payload"], dict):
        return False
    return True


def coerce(record):
    out = dict(record)
    if "payload" not in out:
        out["payload"] = {}
    if "created_at" in out:
        out["created_at"] = float(out["created_at"])
    return out
