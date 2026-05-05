from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np


repo_root = Path(__file__).resolve().parents[1]
python_dir = repo_root / "python"
sys.path.insert(0, str(python_dir))

from training_queue import TrainingQueue


def _producer(
    queue: TrainingQueue,
    producer_id: int,
    batches: int,
    errors: list[Exception],
    errors_lock: threading.Lock,
) -> None:
    try:
        for i in range(batches):
            batch_id = producer_id * 100_000 + i
            X = np.full((8, 8), batch_id, dtype=np.float32)
            y = np.full((8,), batch_id, dtype=np.int32)
            queue.put(X, y)
    except Exception as exc:
        with errors_lock:
            errors.append(exc)


def _consumer(
    queue: TrainingQueue,
    done_event: threading.Event,
    observed: list[int],
    errors: list[Exception],
    errors_lock: threading.Lock,
) -> None:
    try:
        while not done_event.is_set() or queue.qsize() > 0:
            batch = queue.get_batch()
            if batch is None:
                continue
            X, y = batch
            if X.size == 0 or y.size == 0:
                raise AssertionError("Received empty batch")
            x_value = float(X[0, 0])
            if not np.all(X == x_value):
                raise AssertionError("Corrupted X batch")
            if not np.all(y == int(x_value)):
                raise AssertionError("Corrupted y batch")
            observed.append(int(x_value))
    except Exception as exc:
        with errors_lock:
            errors.append(exc)


def test_training_queue_thread_safety() -> None:
    queue = TrainingQueue(max_batches=32)
    batches_per_producer = 200
    done_event = threading.Event()
    observed: list[int] = []
    errors: list[Exception] = []
    errors_lock = threading.Lock()

    consumer = threading.Thread(
        target=_consumer,
        args=(queue, done_event, observed, errors, errors_lock),
    )
    producer_a = threading.Thread(
        target=_producer,
        args=(queue, 1, batches_per_producer, errors, errors_lock),
    )
    producer_b = threading.Thread(
        target=_producer,
        args=(queue, 2, batches_per_producer, errors, errors_lock),
    )

    consumer.start()
    producer_a.start()
    producer_b.start()

    producer_a.join(timeout=5)
    producer_b.join(timeout=5)
    done_event.set()
    consumer.join(timeout=5)

    assert not producer_a.is_alive()
    assert not producer_b.is_alive()
    assert not consumer.is_alive()

    if errors:
        raise errors[0]

    assert len(observed) > 0
    assert len(observed) <= batches_per_producer * 2
