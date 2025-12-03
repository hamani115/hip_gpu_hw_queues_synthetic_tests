#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>

int KERNEL_LAUNCHES = 1000;


inline void hip_check(const char* file, int line, const char* cmd, hipError_t result) {\
    if (__builtin_expect(result == hipSuccess, true))\
        return;\

    std::ostringstream out;\
    out << "\n";\
    out << file << ", line " << line << ":\n";\
    out << "HIP_CHECK(" << cmd << ");\n";\
    out << hipGetErrorName(result) << ": " << hipGetErrorString(result); \
    throw std::runtime_error(out.str()); \
}\

#define HIP_CHECK(ARG) (hip_check(__FILE__, __LINE__, #ARG, (ARG)))\


__global__ void simpleKernel(int *data, int val, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = val + data[idx];
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_streams>" << std::endl;
        return 1;
    }

    int numStreams = std::stoi(argv[1]);
    // 1M elements
    const int arraySize = 1 << 20;
    const int bytes = arraySize * sizeof(int);

    std::cout << "Using " << numStreams << " streams..." << std::endl;

    // Host pinned memory
    std::vector<int*> h_in(numStreams), h_out(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        HIP_CHECK(hipHostMalloc(&h_in[i], bytes));
        HIP_CHECK(hipHostMalloc(&h_out[i], bytes));
        // initialize h_in
        for (int j = 0; j < arraySize; ++j) {
            // h_in[i][j] = (j+1) / 10.0;
            h_in[i][j] = (j+1) % 10;
        }
    }

    // Create hip streams vector
    std::vector<hipStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }

    // Device buffers
    std::vector<int*> d_data(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        HIP_CHECK(hipMallocAsync(&d_data[i], bytes, streams[i]));
    }

    // Sync all streams BEFORE
    for (int i = 0; i < numStreams; ++i) {
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    // START TIMER
    auto start = std::chrono::steady_clock::now();

    // Launch single kernel for each stream
    dim3 blockSize(1024);
    dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x);
    for (int i = 0; i < numStreams; ++i) {

        // H2D copy
        HIP_CHECK(hipMemcpyAsync(d_data[i], h_in[i], bytes, hipMemcpyHostToDevice, streams[i]));

        for (int j = 0; j < KERNEL_LAUNCHES; ++j) {
            simpleKernel<<<gridSize, blockSize, 0, streams[i]>>>(d_data[i], (i+1), arraySize);
        }

        // D2H copy
        HIP_CHECK(hipMemcpyAsync(h_out[i], d_data[i], bytes, hipMemcpyDeviceToHost, streams[i]));
    }

    // Sync all streams AFTER
    for (int i = 0; i < numStreams; ++i) {
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    // END TIMER
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Total execution time (H2D + kernel + D2H across all streams): "
              << ms << " ms" << '\n';

    // Computation Output Check
    // std::cout << "CHECK OUTPUT:\n";
    // for (int i = 0; i < numStreams; ++i) {
    //     // std::streamsize old_prec = std::cout.precision();
    //     // std::ios::fmtflags old_flags = std::cout.flags();
    //     // std::cout << std::fixed << std::setprecision(7);
        
    //     std::cout << "stream " << i << ":\n"
	//               << h_out[i][0] << ", " << h_out[i][1] << ", " << h_out[i][2] << ",... , " << h_out[i][arraySize-2] << ", "<< h_out[i][arraySize-1]
    //               << '\n';

    //     // std::cout.precision(old_prec);
    //     // std::cout.flags(old_flags);
    // }

    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        HIP_CHECK(hipFreeAsync(d_data[i], streams[i]));
        HIP_CHECK(hipStreamDestroy(streams[i]));
    }

    return 0;
}

