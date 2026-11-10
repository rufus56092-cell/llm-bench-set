"""Module cache: LRU cache with TTL."""
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
