"""Module auth: token issue and verification."""
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
