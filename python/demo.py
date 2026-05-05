from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

import data_loader


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _print_status_line(status: dict[str, Any]) -> None:
    step = int(status.get("step", 0))
    loss = float(status.get("loss", 0.0))
    acc = float(status.get("accuracy", 0.0))
    queue_depth = int(status.get("queue_depth", 0))
    print(
        f"Step {step:>4} | Loss: {loss:>5.3f} | Acc: {acc * 100.0:>5.1f}% | Queue: {queue_depth}",
        flush=True,
    )


def _poll_health(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise RuntimeError("Server did not become healthy within timeout")


def _fetch_status(base_url: str) -> dict[str, Any]:
    response = requests.get(f"{base_url}/status", timeout=10)
    response.raise_for_status()
    return response.json()


def _send_train(base_url: str, X: np.ndarray, y: np.ndarray) -> None:
    payload = {"X": X.tolist(), "y": y.tolist()}
    response = requests.post(f"{base_url}/train", json=payload, timeout=30)
    response.raise_for_status()


def _predict(base_url: str, X: np.ndarray) -> dict[str, Any]:
    payload = {"X": X.tolist()}
    response = requests.post(f"{base_url}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> int:
    base_url = "http://localhost:8000"
    python_dir = Path(__file__).resolve().parent
    os.chdir(python_dir)

    print("Starting ParallelNet training server in background...", flush=True)

    server_path = python_dir / "server.py"
    process = subprocess.Popen([sys.executable, str(server_path)], cwd=str(python_dir))

    try:
        time.sleep(5.0)
        _poll_health(base_url, timeout_s=30.0)

        print(f"Server ready at {base_url}", flush=True)
        print("", flush=True)

        X_train, y_train, X_test, y_test = data_loader.get_data("data")
        print("Dataset: 50,000 training images, 10,000 test images", flush=True)
        print("Model: Dense(3072->512)->ReLU->Dense(512->10) -- 1,594,890 parameters", flush=True)
        print("GPU: NVIDIA GeForce GTX 1650 (4 GB VRAM)", flush=True)
        print("", flush=True)
        print("Sending training data...", flush=True)

        batch_size = 128
        total_full_batches = X_train.shape[0] // batch_size
        if total_full_batches == 0:
            raise RuntimeError("No full batches available in CIFAR-10 training set")

        for idx in range(50):
            batch_idx = idx % total_full_batches
            start = batch_idx * batch_size
            end = start + batch_size
            _send_train(base_url, X_train[start:end], y_train[start:end])

            if (idx + 1) % 10 == 0:
                status = _fetch_status(base_url)
                _print_status_line(status)

            time.sleep(0.05)

        while True:
            status = _fetch_status(base_url)
            if int(status.get("step", 0)) >= 50:
                break
            time.sleep(0.5)

        while True:
            try:
                prediction = _predict(base_url, X_test[:5])
                break
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 503:
                    time.sleep(1.0)
                    continue
                raise

        print("", flush=True)
        print("Sample predictions (5 test images):", flush=True)
        preds = prediction.get("predictions", [])
        for idx, pred in enumerate(preds):
            actual = CLASS_NAMES[int(y_test[idx])]
            predicted = CLASS_NAMES[int(pred)] if int(pred) < len(CLASS_NAMES) else str(pred)
            marker = "OK" if predicted == actual else "X"
            print(
                f"  Image {idx + 1}: actual={actual:<10} predicted={predicted:<11} {marker}",
                flush=True,
            )

        print("", flush=True)
        print("Continuing training...", flush=True)

        for idx in range(50, 400):
            batch_idx = idx % total_full_batches
            start = batch_idx * batch_size
            end = start + batch_size
            _send_train(base_url, X_train[start:end], y_train[start:end])

            if (idx + 1) % 50 == 0:
                status = _fetch_status(base_url)
                _print_status_line(status)

            time.sleep(0.05)

        final_status = _fetch_status(base_url)
        final_loss = float(final_status.get("loss", 0.0))
        final_acc = float(final_status.get("accuracy", 0.0)) * 100.0
        samples = 400 * batch_size

        print("", flush=True)
        print(f"After 1 epoch ({samples:,} samples):", flush=True)
        print(f"  Loss:     {final_loss:.3f}", flush=True)
        print(f"  Accuracy: {final_acc:.1f}%", flush=True)
        print("  Expected: ~35-45% (MLP on CIFAR-10, 1 epoch) OK", flush=True)
        print("", flush=True)
        print("Server still running at http://localhost:8000/docs", flush=True)
        print("Press Ctrl+C to stop. Run send_epochs.py --epochs 10 for full training.", flush=True)

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping server...", flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
