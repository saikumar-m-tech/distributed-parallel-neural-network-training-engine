from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np


class TrainingQueue:
    def __init__(self, max_batches: int = 32) -> None:
        self._max_batches = max_batches
        self._queue: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._condition = threading.Condition()

    def put(self, X: np.ndarray, y: np.ndarray) -> None:
        with self._condition:
            if len(self._queue) >= self._max_batches:
                self._queue.popleft()
            self._queue.append((X, y))
            self._condition.notify()

    def get_batch(self) -> tuple[np.ndarray, np.ndarray] | None:
        deadline = time.monotonic() + 0.1
        with self._condition:
            while not self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._queue.popleft()

    def qsize(self) -> int:
        with self._condition:
            return len(self._queue)
