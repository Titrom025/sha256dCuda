#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>
#include <cstring>

// Function to print bytes as a hexadecimal string
std::string to_hex_string(const std::vector<uint8_t>& data) {
    std::stringstream ss;
    for (uint8_t byte : data) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
    }
    return ss.str();
}

// CUDA kernel declaration
extern "C" {
    __global__ void sha256d_kernel(uint32_t* hash, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t c4, uint32_t c5, uint32_t c6, uint32_t c7, uint32_t c8, uint32_t c9, uint32_t c10, uint32_t c11, uint32_t c12, uint32_t c13, uint32_t c14, uint32_t c15);
}

int main() {
    const std::string input_data = "abc";

    uint64_t original_bit_length = input_data.size() * 8;

    size_t pad_len = 64 - ((input_data.size() + 9) % 64);
    std::vector<uint8_t> padded_input_data(input_data.begin(), input_data.end());
    padded_input_data.push_back(0x80);
    padded_input_data.insert(padded_input_data.end(), pad_len, 0x00);
    for (int i = 7; i >= 0; --i) {
        padded_input_data.push_back((original_bit_length >> (i * 8)) & 0xFF);
    }

    std::vector<uint32_t> input_values(16);
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = (padded_input_data[i * 4] << 24) |
                          (padded_input_data[i * 4 + 1] << 16) |
                          (padded_input_data[i * 4 + 2] << 8) |
                          padded_input_data[i * 4 + 3];
    }

    std::vector<uint32_t> output_buffer(8);

    uint32_t* output_gpu;
    cudaMalloc(&output_gpu, output_buffer.size() * sizeof(uint32_t));

    sha256d_kernel<<<1, 1>>>(output_gpu, input_values[0], input_values[1], input_values[2], input_values[3], input_values[4], input_values[5], input_values[6], input_values[7], input_values[8], input_values[9], input_values[10], input_values[11], input_values[12], input_values[13], input_values[14], input_values[15]);

    cudaMemcpy(output_buffer.data(), output_gpu, output_buffer.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<uint8_t> cuda_result_bytes;
    for (uint32_t word : output_buffer) {
        for (int i = 3; i >= 0; --i) {
            cuda_result_bytes.push_back((word >> (i * 8)) & 0xFF);
        }
    }
    std::string cuda_result = to_hex_string(cuda_result_bytes);
    std::cout << "SHA-256d hash (CUDA): " << cuda_result << std::endl;

    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    const EVP_MD* md = EVP_sha256();

    std::vector<uint8_t> input_data_vector(input_data.begin(), input_data.end());
    std::vector<uint8_t> first_hash(EVP_MD_size(md));
    EVP_DigestInit_ex(mdctx, md, NULL);
    EVP_DigestUpdate(mdctx, input_data_vector.data(), input_data_vector.size());
    EVP_DigestFinal_ex(mdctx, first_hash.data(), NULL);

    std::vector<uint8_t> second_hash(EVP_MD_size(md));
    EVP_DigestInit_ex(mdctx, md, NULL);
    EVP_DigestUpdate(mdctx, first_hash.data(), first_hash.size());
    EVP_DigestFinal_ex(mdctx, second_hash.data(), NULL);

    EVP_MD_CTX_free(mdctx);

    std::string correct_result = to_hex_string(second_hash);
    std::cout << "SHA-256d correct (OpenSSL): " << correct_result << std::endl;
    std::cout << "SHA-256d match: " << (cuda_result == correct_result) << std::endl;

    cudaFree(output_gpu);

    return 0;
}