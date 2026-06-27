"""Lightweight in-process performance monitoring.

Tracks latency and throughput for named operations (embedding, search,
generation, end-to-end). Used by the pipeline and surfaced via /health.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any


class MetricsCollector:
    """Accumulates per-operation latency samples and counters."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._lock = threading.Lock()
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._counts: dict[str, int] = defaultdict(int)
        self._max_samples = max_samples
        self._started = time.time()

    def record(self, name: str, seconds: float) -> None:
        with self._lock:
            samples = self._latencies[name]
            samples.append(seconds)
            if len(samples) > self._max_samples:
                del samples[0]
            self._counts[name] += 1

    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start)

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
        return ordered[idx]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            uptime = time.time() - self._started
            ops: dict[str, Any] = {}
            for name, samples in self._latencies.items():
                count = self._counts[name]
                ops[name] = {
                    "count": count,
                    "avg_ms": round(1000 * sum(samples) / len(samples), 2) if samples else 0.0,
                    "p50_ms": round(1000 * self._percentile(samples, 50), 2),
                    "p95_ms": round(1000 * self._percentile(samples, 95), 2),
                    "throughput_per_min": round(count / uptime * 60, 2) if uptime > 0 else 0.0,
                }
            return {"uptime_seconds": round(uptime, 1), "operations": ops}


# Module-level singleton shared across the app.
METRICS = MetricsCollector()
