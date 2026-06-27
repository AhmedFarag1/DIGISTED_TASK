"""Thread-safe LRU + TTL cache used for embeddings and full RAG answers."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable


class LRUCache:
    """Simple capacity-bounded LRU cache with optional per-entry TTL."""

    def __init__(self, capacity: int = 512, ttl_seconds: float | None = None) -> None:
        self.capacity = max(1, capacity)
        self.ttl = ttl_seconds
        self._store: "OrderedDict[Hashable, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _expired(self, ts: float) -> bool:
        return self.ttl is not None and (time.time() - ts) > self.ttl

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self.misses += 1
                return None
            ts, value = item
            if self._expired(ts):
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
