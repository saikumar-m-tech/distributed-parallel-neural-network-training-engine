#ifndef KERNEL_UTILS_CUH
#define KERNEL_UTILS_CUH

#include <cuda_runtime.h>

#if defined(__CUDACC_RTC__) || defined(__CUDA_ARCH__)
#define CUDA_CHECK(call) (call)
#else
#include <stdio.h>
#include <chrono>

/**
 * @brief CUDA error checking macro
 * Prints filename, line number, and CUDA error string on failure
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0) 

/**
 * @brief Get and print CUDA device information
 * Prints device name, compute capability, total VRAM, 
 * shared memory per block, and max threads per block
 */
inline void get_device_info() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("========== CUDA Device Information ==========\n");
    printf("Device ID: %d\n", device);
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", 
           static_cast<float>(prop.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f));
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of MultiProcessors: %d\n", prop.multiProcessorCount);
    printf("============================================\n\n");
}

/**
 * @brief GPU timer class using CUDA events for microsecond precision timing
 * Measures GPU kernel execution time with minimal CPU overhead
 */
class GpuTimer {
private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

public:
    /**
     * @brief Constructor: creates CUDA events
     */
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));
    }

    /**
     * @brief Destructor: destroys CUDA events
     */
    ~GpuTimer() noexcept {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    /**
     * @brief Start timing - record event after synchronizing GPU
     */
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event_, 0));
    }

    /**
     * @brief Stop timing - record event after synchronizing GPU
     */
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_event_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
    }

    /**
     * @brief Get elapsed time in milliseconds
     * @return Elapsed time in milliseconds
     */
    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event_, stop_event_));
        return ms;
    }

    /**
     * @brief Get elapsed time in microseconds
     * @return Elapsed time in microseconds
     */
    float elapsed_us() const {
        return elapsed_ms() * 1000.0f;
    }

    /**
     * @brief Get elapsed time in seconds
     * @return Elapsed time in seconds
     */
    float elapsed_s() const {
        return elapsed_ms() / 1000.0f;
    }
};

template <typename T>
class GpuBuffer {
private:
    T* data_;
    size_t count_;

public:
    GpuBuffer() : data_(nullptr), count_(0) {}

    explicit GpuBuffer(size_t count) : data_(nullptr), count_(count) {
        CUDA_CHECK(cudaMalloc(&data_, sizeof(T) * count_));
    }

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    GpuBuffer(GpuBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                CUDA_CHECK(cudaFree(data_));
            }
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~GpuBuffer() noexcept {
        if (data_ != nullptr) {
            cudaError_t err = cudaFree(data_);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
                fprintf(stderr, "GpuBuffer cudaFree failed: %s\n",
                        cudaGetErrorString(err));
            }
            data_ = nullptr;
        }
    }

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    void copy_from_host(const T* src, size_t count) {
        CUDA_CHECK(cudaMemcpy(data_, src, sizeof(T) * count, cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* dst, size_t count) const {
        CUDA_CHECK(cudaMemcpy(dst, data_, sizeof(T) * count, cudaMemcpyDeviceToHost));
    }

    size_t size() const {
        return count_;
    }
};

#endif // __CUDACC_RTC__

#endif // KERNEL_UTILS_CUH
