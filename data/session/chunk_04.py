"""Module api: HTTP handlers."""
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
