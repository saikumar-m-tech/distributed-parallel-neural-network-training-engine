#ifndef GRADIENT_SYNC_HPP
#define GRADIENT_SYNC_HPP

#ifdef PARALLELNET_NO_MPI

class GradientSync {
public:
	GradientSync() = default;
	~GradientSync() = default;

	int rank() const {
		return 0;
	}

	int world_size() const {
		return 1;
	}

	void allreduce_mean(float* gradient_buffer, int count) {
		(void)gradient_buffer;
		(void)count;
	}
};

#else

#include <mpi.h>

class GradientSync {
public:
	GradientSync() {
		int initialized = 0;
		MPI_Initialized(&initialized);
		if (!initialized) {
			MPI_Init(nullptr, nullptr);
		}
	}

	~GradientSync() {
		int finalized = 0;
		MPI_Finalized(&finalized);
		if (!finalized) {
			MPI_Finalize();
		}
	}

	int rank() const {
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		return rank;
	}

	int world_size() const {
		int size = 1;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		return size;
	}

	void allreduce_mean(float* gradient_buffer, int count) {
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, gradient_buffer, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		int size = world_size();
		if (size <= 1) {
			return;
		}
		for (int i = 0; i < count; ++i) {
			gradient_buffer[i] /= static_cast<float>(size);
		}
	}
};

#endif

#endif // GRADIENT_SYNC_HPP
