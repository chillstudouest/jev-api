"""Request-step stopwatch for System One (logs + response.timings)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class Stopwatch:
    def __init__(self) -> None:
        now = time.perf_counter()
        self._started = now
        self._last = now
        self.steps: dict[str, float] = {}
        self.questions: list[dict[str, str | float]] = []

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        ms = round((now - self._last) * 1000.0, 3)
        self.steps[name] = ms
        self._last = now
        return ms

    def measure(self, name: str, run: Callable[[], T]) -> T:
        started = time.perf_counter()
        result = run()
        self.steps[name] = round((time.perf_counter() - started) * 1000.0, 3)
        self._last = time.perf_counter()
        return result

    def measure_question(self, qid: str, qtype: str, run: Callable[[], T]) -> T:
        started = time.perf_counter()
        result = run()
        ms = round((time.perf_counter() - started) * 1000.0, 3)
        self.questions.append({"id": qid, "type": qtype, "ms": ms})
        self._last = time.perf_counter()
        return result

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000.0, 3)
