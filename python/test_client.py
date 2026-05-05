from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

import data_loader


def _iter_with_progress(total: int) -> tuple[Iterable[int], bool]:
    try:
        from tqdm import tqdm
    except ImportError:
        return range(total), False
    return tqdm(range(total), desc="Sending batches"), True


def _print_status(line: str, use_tqdm: bool) -> None:
    if use_tqdm:
        try:
            from tqdm import tqdm

            tqdm.write(line)
            return
        except ImportError:
            pass
    print(line, flush=True)


def _fetch_status(session: requests.Session, base_url: str) -> dict:
    response = session.get(f"{base_url}/status", timeout=10)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    X_train, y_train, X_test, y_test = data_loader.get_data("data")
    batch_size = 128
    total_full_batches = X_train.shape[0] // batch_size
    batches_to_send = max(0, args.batches)

    if total_full_batches == 0:
        print("No full batches available in CIFAR-10 training set.", flush=True)
        return 1

    base_url = args.server.rstrip("/")

    progress_iter, use_tqdm = _iter_with_progress(batches_to_send)

    with requests.Session() as session:
        for step in progress_iter:
            batch_idx = step % total_full_batches
            start = batch_idx * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            payload = {"X": X_batch.tolist(), "y": y_batch.tolist()}
            response = session.post(f"{base_url}/train", json=payload, timeout=30)
            response.raise_for_status()

            status = _fetch_status(session, base_url)
            line = (
                f"step={status.get('step')} loss={status.get('loss')} "
                f"accuracy={status.get('accuracy')} queue_depth={status.get('queue_depth')}"
            )
            _print_status(line, use_tqdm)

            time.sleep(0.05)

        while True:
            status = _fetch_status(session, base_url)
            queue_depth = int(status.get("queue_depth", 0))
            if queue_depth == 0:
                accuracy = float(status.get("accuracy", 0.0)) * 100.0
                print(
                    f"Training complete. Final accuracy: {accuracy:.1f}%",
                    flush=True,
                )
                break
            time.sleep(2.0)

        predict_payload = {"X": X_test[:10].tolist()}
        response = session.post(f"{base_url}/predict", json=predict_payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        print(f"Predictions: {result.get('predictions')}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
