#include <cstdio>
#include <stdexcept>

#include "../kernels/kernel_utils.cuh"

int main() {
	try {
		GpuBuffer<float> buffer(16);
		throw std::runtime_error("GpuBuffer exception test");
	} catch (const std::exception& ex) {
		std::fprintf(stderr, "Caught exception: %s\n", ex.what());
		return 0;
	}
}
