from __future__ import annotations

import os

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

	config = get_config()
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	world = comm.Get_size()

	X_train, y_train, X_test, y_test = data_loader.get_data(config.data_dir)

	trainer = Trainer(3072, 512, 10, config.learning_rate, rank, world)

	for epoch in range(config.epochs):
		losses = []
		for batch in data_loader.batches(rank, world, config.batch_size, config.data_dir):
			losses.append(trainer.train_step(batch.X, batch.y))

		mean_loss = float(sum(losses) / max(len(losses), 1))
		if rank == 0:
			acc = trainer.get_accuracy(X_test, y_test)
			print(f"Epoch {epoch}: loss={mean_loss:.4f} acc={acc:.2%}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
