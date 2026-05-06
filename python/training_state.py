from __future__ import annotations

import threading
from typing import Any


class TrainingState:
    def __init__(self, ready_steps: int = 50) -> None:
        self._lock = threading.Lock()
        self._ready_steps = max(1, int(ready_steps))
        self.step: int = 0
        self.epoch: int = 0
        self.loss: float = float("inf")
        self.accuracy: float = 0.0
        self.samples_seen: int = 0
        self.is_ready: bool = False
        self.history: list[dict[str, Any]] = []

    def update(self, loss: float, accuracy: float, batch_size: int) -> None:
        with self._lock:
            self.step += 1
            self.loss = float(loss)
            self.accuracy = float(accuracy)
            self.samples_seen += int(batch_size)
            if self.step >= self._ready_steps:
                self.is_ready = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "step": self.step,
                "epoch": self.epoch,
                "loss": self.loss,
                "accuracy": self.accuracy,
                "samples_seen": self.samples_seen,
                "is_ready": self.is_ready,
                "history": [entry.copy() for entry in self.history],
            }

    def add_history_point(self) -> None:
        with self._lock:
            self.history.append(
                {
                    "step": self.step,
                    "loss": self.loss,
                    "accuracy": self.accuracy,
                }
            )
            if len(self.history) > 1000:
                self.history = self.history[-1000:]
