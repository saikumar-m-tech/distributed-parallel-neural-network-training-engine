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

    def get_batch(self, block: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        if not block:
            with self._condition:
                if not self._queue:
                    return None
                return self._queue.popleft()

        deadline = time.monotonic() + 0.1
        with self._condition:
            while not self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._queue.popleft()

    def drain_batch(self, max_batches: int = 8) -> tuple[np.ndarray, np.ndarray] | None:
        max_batches = max(1, int(max_batches))
        first = self.get_batch()
        if first is None:
            return None

        batches = [first]
        for _ in range(max_batches - 1):
            extra = self.get_batch(block=False)
            if extra is None:
                break
            batches.append(extra)

        if len(batches) == 1:
            return first

        X_concat = np.concatenate([batch[0] for batch in batches], axis=0)
        y_concat = np.concatenate([batch[1] for batch in batches], axis=0)
        return X_concat, y_concat

    def qsize(self) -> int:
        with self._condition:
            return len(self._queue)
