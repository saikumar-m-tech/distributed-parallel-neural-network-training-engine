#include <cstdio>
#include <thread>
#include <chrono>

#include "../mpi/gradient_sync.cpp"

int main() {
	GradientSync sync;
	int rank = sync.rank();
	int world = sync.world_size();

	float buffer[3] = {1.0f, 2.0f, 3.0f};
	if (rank == 1) {
		buffer[0] = 3.0f;
		buffer[1] = 4.0f;
		buffer[2] = 5.0f;
	}

	std::this_thread::sleep_for(std::chrono::milliseconds(50 * rank));
	sync.allreduce_mean(buffer, 3);

	std::printf("Rank %d/%d buffer: [%.1f, %.1f, %.1f]\n",
				rank, world, buffer[0], buffer[1], buffer[2]);

	return 0;
}
