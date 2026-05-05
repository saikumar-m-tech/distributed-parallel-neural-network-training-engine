from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import requests

import data_loader


def _fetch_status(session: requests.Session, base_url: str) -> dict:
    response = session.get(f"{base_url}/status", timeout=15)
    response.raise_for_status()
    return response.json()


def _wait_for_queue(session: requests.Session, base_url: str) -> dict:
    while True:
        status = _fetch_status(session, base_url)
        if int(status.get("queue_depth", 0)) == 0:
            return status
        time.sleep(1.0)


def _send_batches(
    session: requests.Session,
    base_url: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for start in range(0, X_shuffled.shape[0], batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        payload = {"X": X_batch.tolist(), "y": y_batch.tolist()}
        response = session.post(f"{base_url}/train", json=payload, timeout=30)
        response.raise_for_status()


def _print_epoch(epoch: int, total: int, loss: float, acc: float, elapsed: float) -> None:
    print(
        f"Epoch {epoch}/{total}: loss={loss:.3f} acc={acc * 100.0:.1f}% time={elapsed:.1f}s",
        flush=True,
    )


def _print_summary(history: list[dict]) -> None:
    print("\nEpoch | Loss  | Accuracy | Time", flush=True)
    print("------+-------+----------+-----", flush=True)
    for row in history:
        epoch = row["epoch"]
        loss = row["loss"]
        acc = row["accuracy"] * 100.0
        elapsed = row["time"]
        print(f"{epoch:>5} | {loss:>5.3f} | {acc:>7.1f}% | {elapsed:>4.0f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    epochs = max(1, args.epochs)
    batch_size = max(1, args.batch_size)
    base_url = args.server.rstrip("/")

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    X_train, y_train, _, _ = data_loader.get_data("data")

    history: list[dict] = []

    with requests.Session() as session:
        for epoch in range(1, epochs + 1):
            start_time = time.perf_counter()
            _send_batches(session, base_url, X_train, y_train, batch_size, seed=123)
            status = _wait_for_queue(session, base_url)
            elapsed = time.perf_counter() - start_time

            loss = float(status.get("loss", 0.0))
            acc = float(status.get("accuracy", 0.0))
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "accuracy": acc,
                    "time": elapsed,
                }
            )
            _print_epoch(epoch, epochs, loss, acc, elapsed)

    _print_summary(history)

    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "training_history.csv"
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("epoch,loss,accuracy,time_seconds\n")
        for row in history:
            handle.write(
                f"{row['epoch']},{row['loss']:.6f},{row['accuracy']:.6f},{row['time']:.3f}\n"
            )

    print("\nExpected results with your GTX 1650:", flush=True)
    print("Epoch 1:  ~35-40% accuracy", flush=True)
    print("Epoch 5:  ~42-47% accuracy", flush=True)
    print("Epoch 10: ~45-52% accuracy", flush=True)
    print("Epoch 20: ~48-55% accuracy (diminishing returns after this)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
