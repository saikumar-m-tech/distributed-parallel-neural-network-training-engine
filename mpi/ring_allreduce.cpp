
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include <mpi.h>

namespace {

void ring_allreduce_sum(float* buffer, int count, MPI_Comm comm) {
	int rank = 0;
	int world = 1;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &world);
	if (world <= 1 || count <= 0) {
		return;
	}

	std::vector<int> counts(world, 0);
	std::vector<int> offsets(world, 0);
	int base = count / world;
	int remainder = count % world;
	int offset = 0;
	int max_count = 0;
	for (int i = 0; i < world; ++i) {
		counts[i] = base + (i < remainder ? 1 : 0);
		offsets[i] = offset;
		offset += counts[i];
		max_count = std::max(max_count, counts[i]);
	}

	std::vector<float> recv_buffer(static_cast<size_t>(max_count), 0.0f);
	int next = (rank + 1) % world;
	int prev = (rank - 1 + world) % world;

	for (int step = 0; step < world - 1; ++step) {
		int send_block = (rank - step + world) % world;
		int recv_block = (rank - step - 1 + world) % world;
		int send_count = counts[send_block];
		int recv_count = counts[recv_block];

		float* send_ptr = buffer + offsets[send_block];
		float* recv_ptr = recv_buffer.data();
		MPI_Sendrecv(
			send_ptr,
			send_count,
			MPI_FLOAT,
			next,
			0,
			recv_ptr,
			recv_count,
			MPI_FLOAT,
			prev,
			0,
			comm,
			MPI_STATUS_IGNORE);

		float* target = buffer + offsets[recv_block];
		for (int i = 0; i < recv_count; ++i) {
			target[i] += recv_ptr[i];
		}
	}

	for (int step = 0; step < world - 1; ++step) {
		int send_block = (rank + 1 - step + world) % world;
		int recv_block = (rank - step + world) % world;
		int send_count = counts[send_block];
		int recv_count = counts[recv_block];

		float* send_ptr = buffer + offsets[send_block];
		float* recv_ptr = buffer + offsets[recv_block];
		MPI_Sendrecv(
			send_ptr,
			send_count,
			MPI_FLOAT,
			next,
			1,
			recv_ptr,
			recv_count,
			MPI_FLOAT,
			prev,
			1,
			comm,
			MPI_STATUS_IGNORE);
	}
}

double benchmark_allreduce(
	const std::vector<float>& base,
	std::vector<float>& scratch,
	MPI_Comm comm,
	int warmup,
	int runs,
	bool use_ring) {
	for (int i = 0; i < warmup; ++i) {
		scratch = base;
		MPI_Barrier(comm);
		if (use_ring) {
			ring_allreduce_sum(scratch.data(), static_cast<int>(scratch.size()), comm);
		} else {
			MPI_Allreduce(MPI_IN_PLACE, scratch.data(), static_cast<int>(scratch.size()),
						MPI_FLOAT, MPI_SUM, comm);
		}
		MPI_Barrier(comm);
	}

	double total = 0.0;
	for (int i = 0; i < runs; ++i) {
		scratch = base;
		MPI_Barrier(comm);
		double start = MPI_Wtime();
		if (use_ring) {
			ring_allreduce_sum(scratch.data(), static_cast<int>(scratch.size()), comm);
		} else {
			MPI_Allreduce(MPI_IN_PLACE, scratch.data(), static_cast<int>(scratch.size()),
						MPI_FLOAT, MPI_SUM, comm);
		}
		MPI_Barrier(comm);
		double end = MPI_Wtime();
		total += (end - start);
	}

	return total / static_cast<double>(std::max(runs, 1));
}

bool validate_sum(const std::vector<float>& buffer, float expected) {
	for (size_t i = 0; i < std::min<size_t>(buffer.size(), 8); ++i) {
		float diff = buffer[i] - expected;
		if (diff < -1e-3f || diff > 1e-3f) {
			return false;
		}
	}
	return true;
}

} // namespace

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int rank = 0;
	int world = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world);

	int runs = 20;
	int warmup = 3;
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
			runs = std::max(1, std::atoi(argv[++i]));
		} else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
			warmup = std::max(0, std::atoi(argv[++i]));
		}
	}

	std::vector<int> sizes = {10000, 100000, 1000000};
	if (rank == 0) {
		std::printf("Ring-AllReduce benchmark (world=%d, runs=%d, warmup=%d)\n",
					world, runs, warmup);
	}

	for (int count : sizes) {
		std::vector<float> base(static_cast<size_t>(count), static_cast<float>(rank + 1));
		std::vector<float> ring_buffer = base;
		std::vector<float> mpi_buffer = base;

		ring_allreduce_sum(ring_buffer.data(), count, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, mpi_buffer.data(), count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		float expected = static_cast<float>(world * (world + 1) / 2.0f);
		if (!validate_sum(ring_buffer, expected) || !validate_sum(mpi_buffer, expected)) {
			if (rank == 0) {
				std::printf("Validation failed for size %d\n", count);
			}
			MPI_Finalize();
			return 1;
		}

		double ring_time = benchmark_allreduce(base, ring_buffer, MPI_COMM_WORLD, warmup, runs, true);
		double mpi_time = benchmark_allreduce(base, mpi_buffer, MPI_COMM_WORLD, warmup, runs, false);

		if (rank == 0) {
			double ring_ms = ring_time * 1000.0;
			double mpi_ms = mpi_time * 1000.0;
			double speedup = mpi_ms / std::max(ring_ms, 1e-9);
			std::printf(
				"Size %7d floats: ring=%8.3f ms, mpi_allreduce=%8.3f ms, speedup=%.2fx\n",
				count,
				ring_ms,
				mpi_ms,
				speedup);
		}
	}

	MPI_Finalize();
	return 0;
}
