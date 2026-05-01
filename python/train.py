from __future__ import annotations

import argparse
import os
import time

from mpi4py import MPI

import data_loader
from config import get_config


def _ensure_cuda_path() -> None:
	if os.name != "nt":
		return
	if "CUDA_PATH" not in os.environ and "CUDA_HOME" not in os.environ:
		return
	cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
	if not cuda_root:
		return
	cuda_bin = os.path.join(cuda_root, "bin")
	if os.path.isdir(cuda_bin):
		os.add_dll_directory(cuda_bin)


def main() -> int:
	_ensure_cuda_path()
	from parallelnet_cpp import Trainer

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch-size", type=int, default=None)
	parser.add_argument("--epochs", type=int, default=None)
	parser.add_argument("--limit-samples", type=int, default=None)
	args = parser.parse_args()

	config = get_config()
	if args.batch_size is not None:
		config = config.__class__(
			data_dir=config.data_dir,
			batch_size=args.batch_size,
			epochs=config.epochs if args.epochs is None else args.epochs,
			learning_rate=config.learning_rate,
			limit_samples=config.limit_samples if args.limit_samples is None else args.limit_samples,
		)
	elif args.epochs is not None or args.limit_samples is not None:
		config = config.__class__(
			data_dir=config.data_dir,
			batch_size=config.batch_size,
			epochs=config.epochs if args.epochs is None else args.epochs,
			learning_rate=config.learning_rate,
			limit_samples=config.limit_samples if args.limit_samples is None else args.limit_samples,
		)
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	world = comm.Get_size()

	X_train, y_train, X_test, y_test = data_loader.get_data(config.data_dir)
	if config.limit_samples is not None:
		X_train = X_train[: config.limit_samples]
		y_train = y_train[: config.limit_samples]

	trainer = Trainer(3072, 512, 10, config.learning_rate, rank, world)

	if rank == 0:
		print(
			f"Config: batch_size={config.batch_size} epochs={config.epochs} lr={config.learning_rate}",
			flush=True,
		)

	total_start = time.perf_counter()

	for epoch in range(config.epochs):
		epoch_start = time.perf_counter()
		losses = []
		samples = 0
		for batch in data_loader.batches(
			rank,
			world,
			config.batch_size,
			config.data_dir,
			limit_samples=config.limit_samples,
		):
			losses.append(trainer.train_step(batch.X, batch.y))
			samples += batch.X.shape[0]

		mean_loss = float(sum(losses) / max(len(losses), 1))
		if rank == 0:
			acc = trainer.get_accuracy(X_test, y_test)
			epoch_time = time.perf_counter() - epoch_start
			throughput = samples / max(epoch_time, 1e-8)
			print(
				f"Epoch {epoch}: loss={mean_loss:.4f} acc={acc:.2%} "
				f"time={epoch_time:.2f}s samples/s={throughput:.1f}",
				flush=True,
			)

	if rank == 0:
		total_time = time.perf_counter() - total_start
		avg_epoch = total_time / max(config.epochs, 1)
		print(
			f"Training done: total_time={total_time:.2f}s avg_epoch={avg_epoch:.2f}s",
			flush=True,
		)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
