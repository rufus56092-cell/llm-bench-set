#!/usr/bin/env python3
"""Generate a neutral multi-file project for the session bench.

Each of the twelve modules is genuinely different code, and each carries exactly
one seeded defect. An earlier version reused a single template for every module:
the model spotted that within one turn ("they are byte-identical copies"), which
is fair but turns the defect count into a measure of the generator rather than
of the model. The runner now also refuses to score a degenerate corpus.

The defects are realistic -- the kind that survive a real review -- and each is
findable by reading the file.

    python runners/make_session_data.py --out data/session
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------- source bodies
MODULES: list[tuple[str, str, str]] = [
("storage", "Persistence layer for records, with atomic write", '''"""Module storage: persistence for records, atomic write."""
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
'''),

("validation", "Field validation for incoming records", '''"""Module validation: field checks for incoming records."""
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
'''),

("auth", "Token issue and verification", '''"""Module auth: token issue and verification."""
from __future__ import annotations

import hashlib
import hmac
import time

SECRET = b"rotate-me"
TTL = 3600


def issue(user_id: str, role: str = "reader") -> str:
    stamp = str(int(time.time()))
    payload = f"{user_id}:{role}:{stamp}"
    sig = hmac.new(SECRET, payload.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{payload}:{sig}"


def verify(token: str):
    parts = token.split(":")
    if len(parts) != 4:
        return None
    user_id, role, stamp, sig = parts
    payload = f"{user_id}:{role}:{stamp}"
    expected = hmac.new(SECRET, payload.encode(), hashlib.sha256).hexdigest()[:16]
    # BUG: plain string comparison leaks timing information. Should use
    # hmac.compare_digest.
    if sig != expected:
        return None
    if int(time.time()) - int(stamp) > TTL:
        return None
    return {"user": user_id, "role": role}


def can_write(claims) -> bool:
    return bool(claims) and claims.get("role") in ("writer", "admin")
'''),

("api", "HTTP handlers and routing", '''"""Module api: HTTP handlers."""
from __future__ import annotations

import storage
import validation

DEFAULT_PAGE = 50


def list_records(query: dict) -> dict:
    # BUG: page_size is taken straight from the query and never capped, so one
    # request can pull the whole store.
    page = int(query.get("page", 1))
    page_size = int(query.get("page_size", DEFAULT_PAGE))
    rows = storage._load()
    start = (page - 1) * page_size
    return {"items": rows[start:start + page_size], "total": len(rows)}


def create_record(body: dict):
    if not validation.validate(body):
        return {"error": "invalid record"}, 400
    storage.put(validation.coerce(body))
    return {"ok": True}, 201


def get_record(record_id: str):
    row = storage.get(record_id)
    if row is None:
        return {"error": "not found"}, 404
    return row, 200
'''),

("cache", "LRU cache with TTL", '''"""Module cache: LRU cache with TTL."""
from __future__ import annotations

import time
from collections import OrderedDict

CAPACITY = 256
TTL = 60.0


class Cache:
    def __init__(self, capacity=CAPACITY, ttl=TTL):
        self.capacity = capacity
        self.ttl = ttl
        self._data = OrderedDict()
        self._stamps = {}

    def get(self, key):
        if key not in self._data:
            return None
        if time.time() - self._stamps[key] > self.ttl:
            self._data.pop(key, None)
            self._stamps.pop(key, None)
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        self._data[key] = value
        self._stamps[key] = time.time()
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            # BUG: mutates the dict while iterating over it. Raises
            # RuntimeError: dictionary changed size during iteration.
            for old in self._data:
                self._data.pop(old, None)
                self._stamps.pop(old, None)
                break

    def clear(self):
        self._data.clear()
        self._stamps.clear()
'''),

("scheduler", "Periodic job runner with retry backoff", '''"""Module scheduler: periodic jobs with retry."""
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
'''),

("metrics", "Counters and timers", '''"""Module metrics: counters and timers."""
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
'''),

("export", "Dump records as CSV or JSON", '''"""Module export: dump records as CSV or JSON."""
from __future__ import annotations

import csv
import io
import json

COLUMNS = ("id", "name", "owner", "created_at")


def to_json(rows) -> str:
    return json.dumps(list(rows), indent=1, ensure_ascii=False)


def to_csv(rows) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(COLUMNS)
    for row in rows:
        # BUG: the line is joined by hand instead of handed to writer.writerow,
        # so a comma inside a name breaks the column layout.
        line = ",".join(str(row.get(c, "")) for c in COLUMNS)
        buf.write(line + "\\n")
    return buf.getvalue()


def to_markdown(rows) -> str:
    head = "| " + " | ".join(COLUMNS) + " |"
    sep = "|" + "---|" * len(COLUMNS)
    body = ["| " + " | ".join(str(r.get(c, "")) for c in COLUMNS) + " |" for r in rows]
    return "\\n".join([head, sep] + body)
'''),

("migrate", "Schema migrations applied in order", '''"""Module migrate: schema migrations applied in order."""
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
'''),

("notify", "Outbound webhook delivery", '''"""Module notify: webhook delivery."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.request

SECRET = b"hook-secret"


def sign(body: bytes) -> str:
    return hmac.new(SECRET, body, hashlib.sha256).hexdigest()


def deliver(url: str, event: dict, timeout=10) -> int:
    body = json.dumps(event).encode()
    headers = {"Content-Type": "application/json"}
    # BUG: the signature covers the body only. The timestamp is sent but not
    # signed, so an old payload can be replayed with a fresh stamp.
    headers["X-Signature"] = sign(body)
    headers["X-Timestamp"] = str(int(time.time()))
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status
'''),

("search", "Substring and field-scoped search", '''"""Module search: substring and field-scoped search."""
from __future__ import annotations


def find(rows, query, field=None):
    q = query.lower()
    hits = []
    for row in rows:
        if field:
            # BUG: the query is lowercased but the stored value is not, so
            # field-scoped search misses any capitalised match.
            value = str(row.get(field, ""))
            if q in value:
                hits.append(row)
        else:
            if any(q in str(v).lower() for v in row.values()):
                hits.append(row)
    return hits


def rank(rows, query):
    q = query.lower()
    scored = []
    for row in rows:
        name = str(row.get("name", "")).lower()
        score = 2 if name.startswith(q) else (1 if q in name else 0)
        if score:
            scored.append((score, row))
    return [r for _, r in sorted(scored, key=lambda t: -t[0])]
'''),

("retention", "Age-based pruning of records and attachments", '''"""Module retention: prune old records and attachments."""
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
'''),
]

DOC_TEMPLATE = """# Document {idx:02d}: {name}

## Purpose
{desc}.

## Contract
Callers pass records as plain dictionaries. The module never mutates the caller's
object; it copies before writing. Reads are not cached at this layer -- the cache
module owns that, and caching here as well was the cause of two stale-read bugs.

## Known limits
- No pagination beyond {page} records in a single call
- Timestamps are stored as floats and compared without rounding
- Failures raise, they do not return sentinel values

## Change history
- Added strict validation on put()
- Removed the implicit default owner
- Split the export helpers out of this module
"""

INCIDENT_TEMPLATE = """Incident {idx:02d}: {title}

Date: 2026-{month:02d}-{day:02d}
Severity: {severity}
What happened: {what}
Symptoms in the log: {symptom}
Root cause: {cause}
Follow-up: {followup}
"""


def build(out: Path, seed: int = 11) -> dict:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    brief = f"""BRIEF -- read this first and remember it, we return to it at the end.

SESSION CODE: AXL-{rng.randint(1000, 9999)}-ROWAN
BASELINE: variant B = {rng.randint(100, 900)}.{rng.randint(10, 99)}, latency delta -{rng.randint(10, 90)}.{rng.randint(10, 99)}%
CLOSED HYPOTHESES (do not re-open): cache stampede, DNS failover, scheduler drift

TODAY'S GOAL: review a {len(MODULES)}-module Python service for defects.
We look for: resource leaks, logic errors, error handling gaps, dead code.
Collect everything you find in findings.md.
"""
    (out / "brief.txt").write_text(brief, encoding="utf-8")
    files.append("brief.txt")

    for i, (name, desc, body) in enumerate(MODULES, 1):
        p = out / f"chunk_{i:02d}.py"
        p.write_text(body, encoding="utf-8")
        files.append(p.name)

    for i, (name, desc, _) in enumerate(MODULES[:8], 1):
        p = out / "docs" / f"doc_{i:02d}.md"
        p.parent.mkdir(exist_ok=True)
        p.write_text(DOC_TEMPLATE.format(idx=i, name=name, desc=desc,
                                         page=rng.choice([100, 250, 500])),
                     encoding="utf-8")
        files.append(f"docs/{p.name}")

    titles = ["write timeout", "stale read after update", "duplicate record",
              "silent auth failure", "export truncated"]
    whats = ["the write path timed out under load",
             "a read returned a record that had just been updated",
             "the same record appeared twice after a retry",
             "requests were rejected with no error in the response",
             "the CSV export stopped mid-file"]
    for i in range(5):
        p = out / "incidents" / f"inc_{i + 1:02d}.txt"
        p.parent.mkdir(exist_ok=True)
        p.write_text(INCIDENT_TEMPLATE.format(
            idx=i + 1, title=titles[i], month=rng.randint(1, 12),
            day=rng.randint(1, 28), severity=rng.choice(["low", "medium", "high"]),
            what=whats[i], symptom=rng.choice(
                ["TimeoutError in write", "assert in handler", "KeyError on owner",
                 "empty response body", "connection reset"]),
            cause="not established" if i % 2 else "partially established",
            followup=rng.choice(["added a metric", "no action", "raised the timeout"])),
            encoding="utf-8")
        files.append(f"incidents/{p.name}")

    meta = {"files": files, "modules": {n: d for n, d, _ in MODULES},
            "brief": brief}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    return {"count": len(files), "dir": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/session")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    info = build(Path(args.out), args.seed)
    print(f"generated {info['count']} files in {info['dir']}")


if __name__ == "__main__":
    main()
