from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from config import get_config
import data_loader
from training_queue import TrainingQueue
from training_state import TrainingState

if TYPE_CHECKING:
    from parallelnet_cpp import Trainer


GPU_NAME = "NVIDIA GeForce GTX 1650"
MICRO_BATCH_SIZE = int(os.environ.get("MICRO_BATCH_SIZE", "8"))

_queue: TrainingQueue | None = None
_state: TrainingState | None = None
_trainer: "Trainer" | None = None
_x_train: np.ndarray | None = None
_y_train: np.ndarray | None = None
_x_test: np.ndarray | None = None
_y_test: np.ndarray | None = None
_worker_thread: threading.Thread | None = None


class TrainRequest(BaseModel):
    X: list[list[float]] | None = None
    y: list[int] | None = None
    indices: list[int] | None = None


class PredictRequest(BaseModel):
    X: list[list[float]]


def _ensure_cuda_path() -> None:
    if os.name != "nt":
        return
    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if not cuda_root:
        return
    cuda_bin = os.path.join(cuda_root, "bin")
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)


def _validate_train_payload(payload: TrainRequest) -> tuple[np.ndarray, np.ndarray]:
    if payload.indices is not None:
        if _x_train is None or _y_train is None:
            raise HTTPException(status_code=503, detail="Training data not initialized")
        indices = np.asarray(payload.indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise HTTPException(status_code=400, detail="indices must be a non-empty list")
        if indices.min() < 0 or indices.max() >= _x_train.shape[0]:
            raise HTTPException(status_code=400, detail="indices out of range")
        X = np.ascontiguousarray(_x_train[indices], dtype=np.float32)
        y = np.ascontiguousarray(_y_train[indices], dtype=np.int32)
        return X, y

    if payload.X is None or payload.y is None:
        raise HTTPException(status_code=400, detail="Provide X/y or indices")

    try:
        X = np.ascontiguousarray(payload.X, dtype=np.float32)
        y = np.ascontiguousarray(payload.y, dtype=np.int32)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid payload data types") from exc
    if X.ndim != 2 or X.shape[1] != 3072:
        raise HTTPException(status_code=400, detail="X must have shape (N, 3072)")
    if y.ndim != 1:
        raise HTTPException(status_code=400, detail="y must have shape (N,)")
    if X.shape[0] != y.shape[0]:
        raise HTTPException(status_code=400, detail="X and y batch sizes must match")
    if X.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Batch size must be > 0")
    return X, y


def _validate_predict_payload(payload: PredictRequest) -> np.ndarray:
    try:
        X = np.ascontiguousarray(payload.X, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid payload data types") from exc
    if X.ndim != 2 or X.shape[1] != 3072:
        raise HTTPException(status_code=400, detail="X must have shape (N, 3072)")
    if X.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Batch size must be > 0")
    return X


def _predict_with_logits(trainer: "Trainer", X: np.ndarray) -> tuple[list[int], list[list[float]]]:
    preds, probs = trainer.predict(X)
    return preds.tolist(), probs.tolist()


def train_worker(
    queue: TrainingQueue,
    state: TrainingState,
    trainer: "Trainer",
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    accuracy_every: int,
) -> None:
    step = 0
    last_accuracy = 0.0
    accuracy_every = max(1, int(accuracy_every))
    micro_batches = max(1, MICRO_BATCH_SIZE)
    while True:
        batch = queue.drain_batch(max_batches=micro_batches)
        if batch is None:
            continue
        X, y = batch
        try:
            loss = trainer.train_step(X, y)
            step += 1
            if step % accuracy_every == 0:
                last_accuracy = trainer.get_accuracy(X_eval, y_eval)
            state.update(loss, last_accuracy, y.shape[0])
            state.add_history_point()
        except Exception as exc:
            print(f"train_worker error: {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queue, _state, _trainer, _x_train, _y_train, _x_test, _y_test, _worker_thread

    _ensure_cuda_path()
    from parallelnet_cpp import Trainer

    config = get_config()
    X_train, y_train, X_test, y_test = data_loader.get_data(config.data_dir)
    accuracy_samples = max(1, int(config.status_accuracy_samples))
    accuracy_samples = min(accuracy_samples, X_test.shape[0])
    X_eval = X_test[:accuracy_samples]
    y_eval = y_test[:accuracy_samples]

    _queue = TrainingQueue(max_batches=config.queue_max_batches)
    _state = TrainingState(ready_steps=config.ready_steps)
    _trainer = Trainer(3072, 512, 10, 0.01, 0, 1)
    _x_train = X_train
    _y_train = y_train
    _x_test = X_test
    _y_test = y_test

    _worker_thread = threading.Thread(
        target=train_worker,
        args=(_queue, _state, _trainer, X_eval, y_eval, config.status_accuracy_every),
        daemon=True,
    )
    _worker_thread.start()

    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_ready() -> tuple[TrainingQueue, TrainingState, "Trainer"]:
    if _queue is None or _state is None or _trainer is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return _queue, _state, _trainer


@app.post("/train")
def train(payload: TrainRequest) -> dict[str, Any]:
    queue, state, _trainer = _require_ready()
    X, y = _validate_train_payload(payload)
    queue.put(X, y)
    snapshot = state.snapshot()
    return {
        "queued": True,
        "queue_depth": queue.qsize(),
        "step": snapshot["step"],
    }


@app.get("/status")
def status() -> dict[str, Any]:
    queue, state, _trainer = _require_ready()
    snapshot = state.snapshot()
    snapshot["queue_depth"] = queue.qsize()
    return snapshot


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    queue, state, trainer = _require_ready()
    snapshot = state.snapshot()
    if not snapshot["is_ready"]:
        raise HTTPException(status_code=503, detail="model not ready yet")
    X = _validate_predict_payload(payload)
    predictions, probabilities = _predict_with_logits(trainer, X)
    return {"predictions": predictions, "probabilities": probabilities}


@app.get("/health")
def health() -> dict[str, Any]:
    queue, state, _trainer = _require_ready()
    snapshot = state.snapshot()
    return {"status": "ok", "gpu": GPU_NAME, "step": snapshot["step"]}


@app.get("/history")
def history() -> dict[str, Any]:
    queue, state, _trainer = _require_ready()
    snapshot = state.snapshot()
    return {"history": snapshot["history"][-100:]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
