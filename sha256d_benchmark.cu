#include <iostream>
#include <vector>
#include <chrono> // For high-resolution clock
#include <cstring> // For strlen
#include <openssl/sha.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

struct HashResult {
  uint64_t nonce;
  uint64_t vcpu;
  uint32_t cpu_id;
};

struct DevHashResult {
  uint64_t nonce;
  uint64_t vcpu, found;
};

extern "C" {
    __global__ void sha256d_kernel(uint64_t start_nonce, DevHashResult *result);
}


void checkCudaError(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}
int main() {
    uint64_t factor = 128;
    uint64_t threads_per_block = 1024;
    uint64_t gpu_threads = 16;

    uint64_t max_iterations = std::numeric_limits<uint64_t>::max();
    uint64_t throughput = (uint64_t)((1U << 19) * factor);
    if (max_iterations < throughput) {
        throughput = max_iterations;
    }

    uint64_t threads_per_gpu_threads_block = threads_per_block * gpu_threads;
    dim3 block(threads_per_block);
    dim3 grid((uint64_t)((throughput + threads_per_gpu_threads_block - 1) / threads_per_gpu_threads_block), gpu_threads);

    uint32_t expired = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1)) + 900;
    uint32_t cpu_id = 0;

    HashResult r;
    r.nonce = UINT64_MAX;
    r.vcpu = UINT64_MAX;
    r.cpu_id = cpu_id;

    DevHashResult *d_result;
    checkCudaError(cudaMalloc(&d_result, sizeof(*d_result)), "cudaMalloc d_result");

    DevHashResult devresult;
    devresult.nonce = UINT64_MAX;
    devresult.vcpu = UINT64_MAX;
    devresult.found = 0;

    checkCudaError(cudaMemcpy(d_result, &devresult, sizeof(devresult), cudaMemcpyHostToDevice), "cudaMemcpy HostToDevice d_result");
    
    uint64_t start_nonce = 1000000;

    uint64_t hashes_computed = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_from_start = start_time - start_time;
    std::chrono::duration<double> total_elapsed(0);
    unsigned iteration_count = 0;

    for (; hashes_computed < max_iterations;) {
        auto start_iteration = std::chrono::high_resolution_clock::now();
        sha256d_kernel<<<grid, block>>>(hashes_computed, d_result);
        checkCudaError(cudaGetLastError(), "Kernel launch");
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        cudaMemcpy(&devresult, d_result, sizeof(devresult), cudaMemcpyDeviceToHost);
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        r.nonce = devresult.nonce;
        r.vcpu = (devresult.vcpu == UINT64_MAX) ? UINT64_MAX : devresult.vcpu;
        if (r.vcpu != UINT64_MAX) {
            printf("r.vcpu: %lu, r.nonce: %lu, \n", r.vcpu, r.nonce);
            printf("devresult.vcpu: %lu, devresult.nonce: %lu, \n", devresult.vcpu, devresult.nonce);
        }

        std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start_iteration;
        hashes_computed += throughput;

        total_elapsed += elapsed;
        iteration_count++;
        // std::cout << "Elapsed time: " << elapsed.count() * 1e3 << " milliseconds (" << elapsed.count() * 1e6 << " microseconds)" << std::endl;
    
        elapsed_from_start = std::chrono::high_resolution_clock::now() - start_time;
        if (elapsed_from_start.count() > 1) {
            break;
        }
    }

    if (r.vcpu != UINT64_MAX) {
        printf("r.vcpu: %lu, r.nonce: %lu, \n", r.vcpu, r.nonce);
        printf("devresult.vcpu: %lu, devresult.nonce: %lu, \n", devresult.vcpu, devresult.nonce);
    } else {
        printf("No solution found!\n");
    }
    
    printf("Throughput: %lu\n", throughput);
    printf("Max iterations: %lu\n", max_iterations);
    printf("Start nonce: %lu\n", start_nonce);
    printf("Block dimensions: (%u, %u, %u)\n", block.x, block.y, block.z);
    printf("Grid dimensions: (%u, %u, %u)\n", grid.x, grid.y, grid.z);

    double kh_per_second = hashes_computed / (elapsed_from_start.count() * 1e3);
    double gh_per_second = hashes_computed / (elapsed_from_start.count() * 1e9);

    std::cout << "Elapsed from start: " << elapsed_from_start.count() << " seconds " << std::endl;
    std::cout << "Hashes calculated: " << hashes_computed << std::endl;
    std::cout << "Performance: " << kh_per_second << " kH/s" << std::endl;
    std::cout << "Performance: " << gh_per_second << " GH/s" << std::endl;

    double average_elapsed = elapsed_from_start.count() / iteration_count;

    std::cout << "Iteration count: " << iteration_count << std::endl;
    std::cout << "Total elapsed time: " << elapsed_from_start.count() << " seconds" << std::endl;
    std::cout << "Average elapsed time per iteration: " << average_elapsed * 1e3 << " ms" << std::endl;

    return 0;
}