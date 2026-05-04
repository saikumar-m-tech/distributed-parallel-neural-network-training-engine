from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")


cp = pytest.importorskip("cupy")


def _load_activation_module() -> "cp.RawModule":
	repo_root = Path(__file__).resolve().parents[1]
	kernels_dir = repo_root / "kernels"
	source_path = kernels_dir / "activations.cu"
	source = source_path.read_text(encoding="utf-8")
	return cp.RawModule(
		code=source,
		options=("--std=c++17", f"-I{kernels_dir}"),
		name_expressions=(
			"relu_forward",
			"relu_backward",
			"softmax_forward",
			"cross_entropy_loss",
		),
	)


def _softmax_reference(x: np.ndarray) -> np.ndarray:
	shifted = x - np.max(x, axis=1, keepdims=True)
	exp = np.exp(shifted)
	return exp / np.sum(exp, axis=1, keepdims=True)


def _load_sgd_module() -> "cp.RawModule":
	source_path = Path(__file__).resolve().parents[1] / "kernels" / "sgd_update.cu"
	source = source_path.read_text(encoding="utf-8")
	return cp.RawModule(
		code=source,
		options=("--std=c++17",),
		name_expressions=("sgd_update",),
	)


@pytest.fixture(scope="module")
def activation_module():
	return _load_activation_module()


@pytest.fixture(scope="module")
def sgd_module():
	return _load_sgd_module()


def test_relu_forward(activation_module):
	x = np.array([-2.0, 0.0, 3.0, -1.0], dtype=np.float32)
	x_gpu = cp.asarray(x)

	block = (256, 1, 1)
	grid = (1, 1, 1)
	kernel = activation_module.get_function("relu_forward")
	kernel(grid, block, (x_gpu, np.int32(x.size)))

	result = cp.asnumpy(x_gpu)
	ref = np.array([0.0, 0.0, 3.0, 0.0], dtype=np.float32)
	assert np.allclose(result, ref, rtol=1e-6, atol=1e-6)


def test_relu_backward(activation_module):
	n = 1000
	x_pre = np.random.randn(n).astype(np.float32)
	dx = np.random.randn(n).astype(np.float32)

	x_pre_gpu = cp.asarray(x_pre)
	dx_gpu = cp.asarray(dx)

	block = (256, 1, 1)
	grid = (math.ceil(n / block[0]), 1, 1)
	kernel = activation_module.get_function("relu_backward")
	kernel(grid, block, (x_pre_gpu, dx_gpu, np.int32(n)))

	result = cp.asnumpy(dx_gpu)
	ref = np.where(x_pre > 0.0, dx, 0.0)
	assert np.allclose(result, ref, rtol=1e-6, atol=1e-6)


def test_softmax_forward(activation_module):
	batch_size = 32
	num_classes = 10

	x = np.random.randn(batch_size, num_classes).astype(np.float32)
	in_gpu = cp.asarray(x)
	out_gpu = cp.empty_like(in_gpu)

	block = (256, 1, 1)
	grid = (batch_size, 1, 1)
	shared_bytes = block[0] * 4
	kernel = activation_module.get_function("softmax_forward")
	kernel(grid, block, (in_gpu, out_gpu, np.int32(batch_size), np.int32(num_classes)),
		   shared_mem=shared_bytes)

	result = cp.asnumpy(out_gpu)
	ref = scipy.special.softmax(x, axis=1)
	assert np.allclose(result, ref, rtol=1e-5, atol=1e-6)
	assert np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)


def test_cross_entropy_loss(activation_module):
	batch_size = 256
	num_classes = 64

	logits = np.random.randn(batch_size, num_classes).astype(np.float32)
	probs = _softmax_reference(logits)
	labels = np.random.randint(0, num_classes, size=batch_size).astype(np.int32)

	probs_gpu = cp.asarray(probs)
	labels_gpu = cp.asarray(labels)
	loss_gpu = cp.zeros((1,), dtype=cp.float32)

	block = (256, 1, 1)
	grid = (math.ceil(batch_size / block[0]), 1, 1)
	shared_bytes = block[0] * 4
	kernel = activation_module.get_function("cross_entropy_loss")
	kernel(
		grid,
		block,
		(probs_gpu, labels_gpu, loss_gpu, np.int32(batch_size), np.int32(num_classes)),
		shared_mem=shared_bytes,
	)

	result = float(cp.asnumpy(loss_gpu)[0])
	ref = float(np.mean(-np.log(probs[np.arange(batch_size), labels])))
	assert np.allclose(result, ref, rtol=1e-5, atol=1e-6)


def test_sgd_update(sgd_module):
	n = 2048
	weights = np.full(n, 1.0, dtype=np.float32)
	gradients = np.full(n, 0.1, dtype=np.float32)
	learning_rate = np.float32(0.01)

	block = (256, 1, 1)
	grid = (math.ceil(n / block[0]), 1, 1)
	kernel = sgd_module.get_function("sgd_update")
	weights_gpu = cp.asarray(weights)
	gradients_gpu = cp.asarray(gradients)

	kernel(grid, block, (weights_gpu, gradients_gpu, learning_rate, np.int32(n)))
	cp.cuda.runtime.deviceSynchronize()

	result = cp.asnumpy(weights_gpu)
	ref = np.full(n, 0.999, dtype=np.float32)
	assert np.allclose(result, ref, rtol=1e-6, atol=1e-6)

	kernel(grid, block, (weights_gpu, gradients_gpu, learning_rate, np.int32(n)))
	cp.cuda.runtime.deviceSynchronize()

	result = cp.asnumpy(weights_gpu)
	ref = np.full(n, 0.998, dtype=np.float32)
	assert np.allclose(result, ref, rtol=1e-6, atol=1e-6)

	large_n = 100_000
	weights_large = np.full(large_n, 2.0, dtype=np.float32)
	gradients_large = np.full(large_n, 0.5, dtype=np.float32)
	weights_large_gpu = cp.asarray(weights_large)
	gradients_large_gpu = cp.asarray(gradients_large)
	large_grid = (math.ceil(large_n / block[0]), 1, 1)

	kernel(
		large_grid,
		block,
		(weights_large_gpu, gradients_large_gpu, learning_rate, np.int32(large_n)),
	)
	cp.cuda.runtime.deviceSynchronize()

	result = cp.asnumpy(weights_large_gpu)
	ref = np.full(large_n, 1.995, dtype=np.float32)
	assert np.allclose(result, ref, rtol=1e-6, atol=1e-6)
