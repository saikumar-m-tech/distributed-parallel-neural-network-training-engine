from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

import data_loader


def _fetch_status(session: requests.Session, base_url: str) -> dict[str, Any]:
    response = session.get(f"{base_url}/status", timeout=10)
    response.raise_for_status()
    return response.json()


def _send_indices(session: requests.Session, base_url: str, indices: list[int]) -> None:
    response = session.post(f"{base_url}/train", json={"indices": indices}, timeout=30)
    response.raise_for_status()


def _predict(session: requests.Session, base_url: str, X: np.ndarray) -> list[int] | None:
    response = session.post(f"{base_url}/predict", json={"X": X.tolist()}, timeout=30)
    if response.status_code == 503:
        return None
    response.raise_for_status()
    payload = response.json()
    return payload.get("predictions", [])


def _worker_loop(
    worker_id: int,
    base_url: str,
    num_samples: int,
    batch_size: int,
    status_every: int,
    queue_threshold: int,
    stop_event: threading.Event,
    counter: dict[str, int],
    counter_lock: threading.Lock,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed + worker_id)
    with requests.Session() as session:
        sent = 0
        while not stop_event.is_set():
            indices = rng.integers(0, num_samples, size=batch_size).tolist()
            _send_indices(session, base_url, indices)
            sent += 1
            if status_every > 0 and sent % status_every == 0:
                try:
                    status = _fetch_status(session, base_url)
                except requests.RequestException:
                    time.sleep(0.05)
                else:
                    if int(status.get("queue_depth", 0)) >= queue_threshold:
                        time.sleep(0.01)

            with counter_lock:
                counter["batches"] += 1


def _eval_accuracy(
    session: requests.Session,
    base_url: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> float | None:
    if batch_size <= 0:
        return None
    batch_size = min(batch_size, X_test.shape[0])
    indices = rng.choice(X_test.shape[0], size=batch_size, replace=False)
    preds = _predict(session, base_url, X_test[indices])
    if preds is None:
        return None
    if len(preds) != batch_size:
        return None
    return float(np.mean(np.asarray(preds) == y_test[indices]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--run-seconds", type=int, default=120)
    parser.add_argument("--accuracy-interval", type=int, default=15)
    parser.add_argument("--predict-batch", type=int, default=128)
    parser.add_argument("--status-every", type=int, default=10)
    parser.add_argument("--queue-threshold", type=int, default=96)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    base_url = args.server.rstrip("/")
    workers = max(1, args.workers)
    batch_size = max(1, args.batch_size)
    run_seconds = max(1, args.run_seconds)
    accuracy_interval = max(1, args.accuracy_interval)
    predict_batch = max(1, args.predict_batch)
    status_every = max(0, args.status_every)
    queue_threshold = max(1, args.queue_threshold)

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    X_train, _, X_test, y_test = data_loader.get_data("data")

    stop_event = threading.Event()
    counter_lock = threading.Lock()
    counter = {"batches": 0}

    threads: list[threading.Thread] = []
    for worker_id in range(workers):
        thread = threading.Thread(
            target=_worker_loop,
            args=(
                worker_id,
                base_url,
                X_train.shape[0],
                batch_size,
                status_every,
                queue_threshold,
                stop_event,
                counter,
                counter_lock,
                args.seed,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    rng = np.random.default_rng(args.seed + 999)
    with requests.Session() as session:
        start_time = time.perf_counter()
        next_accuracy = start_time + accuracy_interval
        try:
            while time.perf_counter() - start_time < run_seconds:
                time.sleep(0.2)
                now = time.perf_counter()
                if now >= next_accuracy:
                    acc = _eval_accuracy(session, base_url, X_test, y_test, predict_batch, rng)
                    try:
                        status = _fetch_status(session, base_url)
                    except requests.RequestException:
                        status = {}

                    step = int(status.get("step", 0))
                    loss = float(status.get("loss", 0.0))
                    train_acc = float(status.get("accuracy", 0.0))
                    queue_depth = int(status.get("queue_depth", 0))
                    pred_msg = "n/a" if acc is None else f"{acc * 100.0:.1f}%"
                    print(
                        " | ".join(
                            [
                                f"Step {step}",
                                f"Loss {loss:.3f}",
                                f"TrainAcc {train_acc * 100.0:.1f}%",
                                f"PredAcc {pred_msg}",
                                f"Queue {queue_depth}",
                            ]
                        ),
                        flush=True,
                    )
                    next_accuracy += accuracy_interval
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()

    for thread in threads:
        thread.join(timeout=2.0)

    with counter_lock:
        total_batches = counter["batches"]
    duration = max(1e-6, time.perf_counter() - start_time)
    print(
        f"Sent {total_batches} batches in {duration:.1f}s"
        f" ({total_batches / duration:.1f} batches/s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
