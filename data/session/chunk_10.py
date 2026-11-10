"""Module notify: webhook delivery."""
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
