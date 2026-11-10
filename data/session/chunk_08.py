"""Module export: dump records as CSV or JSON."""
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
        buf.write(line + "\n")
    return buf.getvalue()


def to_markdown(rows) -> str:
    head = "| " + " | ".join(COLUMNS) + " |"
    sep = "|" + "---|" * len(COLUMNS)
    body = ["| " + " | ".join(str(r.get(c, "")) for c in COLUMNS) + " |" for r in rows]
    return "\n".join([head, sep] + body)
