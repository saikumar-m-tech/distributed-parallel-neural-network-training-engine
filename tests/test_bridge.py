import os
import pathlib

import numpy as np
import pytest


if os.name == "nt":
	cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
	if cuda_path:
		cuda_bin = os.path.join(cuda_path, "bin")
		if os.path.isdir(cuda_bin):
			os.add_dll_directory(cuda_bin)

parallelnet_cpp = pytest.importorskip("parallelnet_cpp")


def test_trainer_loss_decreases(tmp_path: pathlib.Path):
	trainer = parallelnet_cpp.Trainer(784, 128, 10, 0.01, 0, 1)
	batch = 16
	x = np.zeros((batch, 784), dtype=np.float32)
	y = np.zeros((batch,), dtype=np.int32)

	first = trainer.train_step(x, y)
	assert isinstance(first, float)

	losses = [first]
	for _ in range(4):
		losses.append(trainer.train_step(x, y))

	assert losses[-1] < losses[0]

	acc_before = trainer.get_accuracy(x, y)
	weights_path = tmp_path / "weights.bin"
	trainer.save_weights(str(weights_path))
	assert weights_path.exists()

	trainer.train_step(x, y)
	trainer.load_weights(str(weights_path))
	acc_after = trainer.get_accuracy(x, y)
	assert np.isclose(acc_before, acc_after, rtol=0.0, atol=1e-6)
