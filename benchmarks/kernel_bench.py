import math
import time

import cupy as cp
import matplotlib.pyplot as plt


KERNEL_CODE = r"""
#define TILE_SIZE 16

extern "C" __global__ void naive_matmul_kernel(const float* A, const float* B, float* C,
												int M, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < M && col < N) {
		float sum = 0.0f;
		for (int k = 0; k < K; ++k) {
			sum += A[row * K + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}
}

extern "C" __global__ void tiled_matmul_kernel(const float* A, const float* B, float* C,
												int M, int N, int K) {
	__shared__ float tileA[TILE_SIZE][TILE_SIZE];
	__shared__ float tileB[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	float sum = 0.0f;
	int tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

	for (int t = 0; t < tiles; ++t) {
		int a_col = t * TILE_SIZE + threadIdx.x;
		int b_row = t * TILE_SIZE + threadIdx.y;

		if (row < M && a_col < K) {
			tileA[threadIdx.y][threadIdx.x] = A[row * K + a_col];
		} else {
			tileA[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if (b_row < K && col < N) {
			tileB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
		} else {
			tileB[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k) {
			sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < M && col < N) {
		C[row * N + col] = sum;
	}
}
"""


def gflops_for_gemm(m, n, k, seconds):
	ops = 2.0 * m * n * k
	return ops / (seconds * 1e9)


def benchmark_kernel(kernel, a, b, c, m, n, k, block, grid, warmup=2, runs=5):
	for _ in range(warmup):
		kernel(grid, block, (a, b, c, m, n, k))
	cp.cuda.Stream.null.synchronize()

	start = cp.cuda.Event()
	end = cp.cuda.Event()
	start.record()
	for _ in range(runs):
		kernel(grid, block, (a, b, c, m, n, k))
	end.record()
	end.synchronize()
	elapsed_ms = cp.cuda.get_elapsed_time(start, end)
	return (elapsed_ms / 1000.0) / runs


def benchmark_cublas(a, b, warmup=2, runs=5):
	for _ in range(warmup):
		_ = a @ b
	cp.cuda.Stream.null.synchronize()

	start = cp.cuda.Event()
	end = cp.cuda.Event()
	start.record()
	for _ in range(runs):
		_ = a @ b
	end.record()
	end.synchronize()
	elapsed_ms = cp.cuda.get_elapsed_time(start, end)
	return (elapsed_ms / 1000.0) / runs


def main():
	sizes = [64, 128, 256, 512, 1024]
	module = cp.RawModule(code=KERNEL_CODE)
	naive_kernel = module.get_function("naive_matmul_kernel")
	tiled_kernel = module.get_function("tiled_matmul_kernel")

	naive_gflops = []
	tiled_gflops = []
	cublas_gflops = []

	print("Benchmarking CUDA matmul kernels (float32)\n")

	for size in sizes:
		m = n = k = size
		a = cp.random.random((m, k), dtype=cp.float32)
		b = cp.random.random((k, n), dtype=cp.float32)
		c = cp.empty((m, n), dtype=cp.float32)

		block = (16, 16, 1)
		grid = (math.ceil(n / 16), math.ceil(m / 16), 1)

		naive_time = benchmark_kernel(naive_kernel, a, b, c, m, n, k, block, grid)
		tiled_time = benchmark_kernel(tiled_kernel, a, b, c, m, n, k, block, grid)
		cublas_time = benchmark_cublas(a, b)

		naive_perf = gflops_for_gemm(m, n, k, naive_time)
		tiled_perf = gflops_for_gemm(m, n, k, tiled_time)
		cublas_perf = gflops_for_gemm(m, n, k, cublas_time)

		naive_gflops.append(naive_perf)
		tiled_gflops.append(tiled_perf)
		cublas_gflops.append(cublas_perf)

		percent = (tiled_perf / cublas_perf) * 100.0 if cublas_perf > 0 else 0.0
		print(f"Size {size:4d}: tiled_matmul is {percent:6.2f}% of cuBLAS")

	plt.figure(figsize=(8, 5))
	plt.plot(sizes, naive_gflops, marker="o", label="naive_matmul")
	plt.plot(sizes, tiled_gflops, marker="o", label="tiled_matmul")
	plt.plot(sizes, cublas_gflops, marker="o", label="cuBLAS SGEMM")
	plt.title("GEMM Performance (GFLOPs)")
	plt.xlabel("Matrix size (N = M = K)")
	plt.ylabel("GFLOPs")
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.legend()
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
