// Matrix multiplication kernels and reference CPU implementation.
#include <cuda_runtime.h>

#include "kernel_utils.cuh"

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

void naive_matmul_gpu(const float* A, const float* B, float* C,
					  int M, int N, int K) {
	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
	naive_matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
	CUDA_CHECK(cudaGetLastError());
}

void tiled_matmul_gpu(const float* A, const float* B, float* C,
					  int M, int N, int K) {
	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
	tiled_matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
	CUDA_CHECK(cudaGetLastError());
}

void matmul_gpu(const float* A, const float* B, float* C,
				int M, int N, int K) {
	tiled_matmul_gpu(A, B, C, M, N, K);
}

void matmul_cpu(const float* A, const float* B, float* C,
				int M, int N, int K) {
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float sum = 0.0f;
			for (int k = 0; k < K; ++k) {
				sum += A[i * K + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}
