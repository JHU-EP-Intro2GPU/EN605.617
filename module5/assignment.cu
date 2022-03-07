#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include "benchmarking.h"
#include "assignment.h"

#define MAX_N_INTS 512

using u32 = unsigned int;

///////////////////////////////////////////////////////////////////////////////
// Constant memory
// __constant__ static const u32 fofo = 0xF0F0;
// __constant__ static const u32 ofof = 0x0F0F;
__constant__ u32 c_array1[MAX_N_INTS];
__constant__ u32 c_array2[MAX_N_INTS];

template<typename T>
__global__
void add_matrices_constant(T dest_matrix[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = c_array1[i] + c_array2[i];
}

template<typename T>
__global__
void subtract_matrices_constant(T dest_matrix[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = c_array1[i] - c_array2[i];
}

template<typename T>
__global__
void multiply_matrices_constant(T dest_matrix[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = c_array1[i] * c_array2[i];
}

template<typename T>
__global__
void modulo_matrices_constant(T dest_matrix[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = c_array1[i] % c_array2[i];
}


///////////////////////////////////////////////////////////////////////////////
// Functions using shared memory
template<typename T>
__global__
void add_matrices_shared(T dest_matrix[],
	T matrix_i[], T matrix_j[])
{
    __shared__ u32 s_array1[MAX_N_INTS];
    __shared__ u32 s_array2[MAX_N_INTS];
    
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_array1[i] = matrix_i[i];
    s_array2[i] = matrix_j[i];
    dest_matrix[i] = s_array1[i] + s_array2[i];
}

// Subtract two 1-d arrays of Ts together
template<typename T>
__global__
void subtract_matrices_shared(T dest_matrix[],
	T matrix_i[], T matrix_j[])
{
    __shared__ u32 s_array1[MAX_N_INTS];
    __shared__ u32 s_array2[MAX_N_INTS];
    
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_array1[i] = matrix_i[i];
    s_array2[i] = matrix_j[i];
    dest_matrix[i] = s_array1[i] - s_array2[i];
}

// Multiply two 1-d arrays of Ts together
template<typename T>
__global__
void multiply_matrices_shared(T dest_matrix[],
	T matrix_i[], T matrix_j[])
{
    __shared__ u32 s_array1[MAX_N_INTS];
    __shared__ u32 s_array2[MAX_N_INTS];
    
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_array1[i] = matrix_i[i];
    s_array2[i] = matrix_j[i];
    dest_matrix[i] = s_array1[i] * s_array2[i];
}

// Modulo divide two 1-d arrays of Ts together
template<typename T>
__global__
void modulo_matrices_shared(T dest_matrix[],
	T matrix_i[], T matrix_j[])
{
    __shared__ u32 s_array1[MAX_N_INTS];
    __shared__ u32 s_array2[MAX_N_INTS];

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_array1[i] = matrix_i[i];
    s_array2[i] = matrix_j[i];
    dest_matrix[i] = s_array1[i] % s_array2[i];
}

///////////////////////////////////////////////////////////////////////////////

void run_4_kernels_constant(u32 * results, const u32 n_blocks,
const u32 block_size) {
    add_matrices_constant<<<n_blocks, block_size>>>(results);
    subtract_matrices_constant<<<n_blocks, block_size>>>(results);
    multiply_matrices_constant<<<n_blocks, block_size>>>(results);
    modulo_matrices_constant<<<n_blocks, block_size>>>(results);
}

void run_4_kernels_shared(u32 * results, u32 *  data1, u32 * data2,
const u32 n_blocks, const u32 block_size) {
    add_matrices_shared<<<n_blocks, block_size>>>(results, data1, data2);
    subtract_matrices_shared<<<n_blocks, block_size>>>(results, data1, data2);
    multiply_matrices_shared<<<n_blocks, block_size>>>(results, data1, data2);
    modulo_matrices_shared<<<n_blocks, block_size>>>(results, data1, data2);
}

void run_constant(u32 * results, u32 *  data1, u32 * data2,
const u32 n_blocks, const u32 block_size, const size_t array_size) {
    // Allocate results buffer
    u32 * device_results;
    cudaMallocHost((void **)&device_results, array_size * sizeof(u32));

    // Copy to constant arrays
    cudaMemcpyToSymbol(c_array1, data1,
        array_size * sizeof(u32));
    cudaMemcpyToSymbol(c_array2, data2,
        array_size * sizeof(u32));

    run_4_kernels_constant(device_results, n_blocks, block_size);

    cudaMemcpy(results, device_results,
        array_size * sizeof(u32), cudaMemcpyDeviceToHost);

    cudaFree(device_results);
}

void run_shared(u32 * results, u32 *  data1, u32 * data2,
const u32 n_blocks, const u32 block_size, const size_t array_size) {
    // Allocate results buffer
    u32 * device_results;
    cudaMallocHost((void **)&device_results, array_size * sizeof(u32));
    u32 * arr1;
    cudaMallocHost((void **)&arr1, array_size * sizeof(u32));
    u32 * arr2;
    cudaMallocHost((void **)&arr2, array_size * sizeof(u32));
    
    // Copy data1 memory to GPU memory
    cudaMemcpy(arr1, data1,
        array_size * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(arr2, data2,
        array_size * sizeof(u32), cudaMemcpyHostToDevice);

    run_4_kernels_shared(device_results, arr1, arr2, n_blocks, block_size);

    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(device_results);
}

int main(int argc, char * argv[]) {
    // Parse command line
    unsigned int block_size = 0; // Threads per block
	unsigned int n_threads = 0; // Total threads we want
	unsigned int n_blocks = 0; // Number of blocks to hold all the threads
    
    if (argc == 3) {
        n_threads = std::stol(std::string(argv[1]));
		block_size = std::stol(std::string(argv[2]));
		n_blocks = (n_threads / block_size) > 0 ? n_threads / block_size : 1;
    }
    else {
        std::cout << "Usage: " << argv[0] << " [block size] [number of threads per block]" << std::endl;
		return 0;
    }

    // Allocate host memory ones vectors for 2 separate runs
    std::vector<u32> ones(MAX_N_INTS, 1);
    std::vector<u32> twos(MAX_N_INTS, 2);

    // Allocate two host destination vectors
    std::vector<u32> dest(MAX_N_INTS);

    // Run constant memory 4 kernels
    TIC();
    run_constant(dest.data(), ones.data(), twos.data(), n_blocks, block_size, MAX_N_INTS);
    std::cout << "Constant mem took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;
    print_vector(dest);

    // Run shared memory 4 kernels
    TIC();
    run_shared(dest.data(), ones.data(), twos.data(), n_blocks, block_size, MAX_N_INTS);
    std::cout << "Shared mem took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;
    print_vector(dest);
}
