"""Module search: substring and field-scoped search."""
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
