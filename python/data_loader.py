from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator, Tuple
from urllib.error import HTTPError, URLError

import numpy as np

try:
	from torchvision.datasets import CIFAR10
	from torchvision.datasets.utils import download_and_extract_archive
except ImportError as exc:
	raise SystemExit("torchvision is required. Install with: pip install torchvision") from exc


@dataclass(frozen=True)
class Batch:
	X: np.ndarray
	y: np.ndarray


_CACHE = None


def shard_indices(total: int, rank: int, world: int) -> Tuple[int, int]:
	chunk = total // world
	start = rank * chunk
	end = (rank + 1) * chunk
	return start, end


def _normalize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	# Normalize per-channel using training statistics.
	mean = train.mean(axis=(0, 1, 2), keepdims=True)
	std = train.std(axis=(0, 1, 2), keepdims=True)
	train = (train - mean) / std
	test = (test - mean) / std
	return train, test


def _flatten(images: np.ndarray) -> np.ndarray:
	# Convert to NCHW layout before flattening to keep channels grouped.
	images = images.transpose(0, 3, 1, 2)
	return images.reshape(images.shape[0], -1)


def _download_cifar10(data_dir: str) -> None:
	# Allow override via CIFAR10_MIRRORS (semicolon-separated URLs).
	mirror_env = os.environ.get("CIFAR10_MIRRORS", "").strip()
	urls = [u for u in mirror_env.split(";") if u] if mirror_env else [CIFAR10.url]
	last_error: Exception | None = None
	for url in urls:
		try:
			download_and_extract_archive(
				url,
				data_dir,
				filename=CIFAR10.filename,
				md5=CIFAR10.tgz_md5,
			)
			return
		except Exception as exc:
			last_error = exc
	if last_error:
		raise last_error


def _resolve_cifar_root(data_dir: str) -> str:
	root = Path(data_dir)
	if (root / "cifar-10-batches-py").exists():
		return str(root)
	archive_root = Path(__file__).resolve().parents[1] / "archive"
	if (archive_root / "cifar-10-batches-py").exists():
		return str(archive_root)
	return str(root)


def load_cifar10(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	root = _resolve_cifar_root(data_dir)
	try:
		train_ds = CIFAR10(root=root, train=True, download=True)
		test_ds = CIFAR10(root=root, train=False, download=True)
	except (HTTPError, URLError):
		_download_cifar10(root)
		train_ds = CIFAR10(root=root, train=True, download=False)
		test_ds = CIFAR10(root=root, train=False, download=False)

	X_train = train_ds.data.astype(np.float32) / 255.0
	y_train = np.asarray(train_ds.targets, dtype=np.int32)
	X_test = test_ds.data.astype(np.float32) / 255.0
	y_test = np.asarray(test_ds.targets, dtype=np.int32)

	X_train, X_test = _normalize(X_train, X_test)
	X_train = _flatten(X_train).astype(np.float32)
	X_test = _flatten(X_test).astype(np.float32)

	return X_train, y_train, X_test, y_test


def get_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	global _CACHE
	if _CACHE is None:
		_CACHE = load_cifar10(data_dir)
	return _CACHE


def batches(
	rank: int,
	world: int,
	batch_size: int,
	data_dir: str,
	limit_samples: int | None = None,
) -> Iterator[Batch]:
	X_train, y_train, _, _ = get_data(data_dir)
	if limit_samples is not None:
		X_train = X_train[:limit_samples]
		y_train = y_train[:limit_samples]
	start, end = shard_indices(X_train.shape[0], rank, world)
	X_shard = X_train[start:end]
	y_shard = y_train[start:end]

	for idx in range(0, X_shard.shape[0], batch_size):
		batch_X = X_shard[idx: idx + batch_size]
		batch_y = y_shard[idx: idx + batch_size]
		yield Batch(batch_X, batch_y)
