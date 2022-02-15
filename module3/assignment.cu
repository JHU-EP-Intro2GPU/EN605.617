#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include "assignment.h"
#include "benchmarking.h"

#define ARRAY_SIZE 32 // CPU array sizes

#define N_THREADS 32 // Number of threads we want in parallel
#define BLOCK_SIZE 32 // Threads per block
#define N_BLOCKS N_THREADS/BLOCK_SIZE // Calculate how many blocks will execute

#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE)) // for mallocs

static std::vector<unsigned int> cpu_array_thread_numbers(ARRAY_SIZE);
static std::vector<unsigned int> cpu_array_random_numbers(ARRAY_SIZE);
static std::vector<unsigned int> results(ARRAY_SIZE);

////////////////////////////////////////////////////////////////////////////////
// Cuda kernels

// Add two 1-d arrays of unsigned ints together 
__global__
void add_matrices(unsigned int dest_matrix[],
	unsigned int matrix_i[], unsigned int matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// dest_matrix[i] = threadIdx.x + matrix_j[i];
    dest_matrix[i] = matrix_i[i] + matrix_j[i];
}

// Subtract two 1-d arrays of unsigned ints together 
__global__
void subtract_matrices(unsigned int dest_matrix[],
	unsigned int matrix_i[], unsigned int matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = matrix_i[i] - matrix_j[i];
}

// Multiply two 1-d arrays of unsigned ints together 
__global__
void multiply_matrices(unsigned int dest_matrix[],
	unsigned int matrix_i[], unsigned int matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = matrix_i[i] * matrix_j[i];
}

// Modulo divide two 1-d arrays of unsigned ints together 
__global__
void modulo_matrices(unsigned int dest_matrix[],
	unsigned int matrix_i[], unsigned int matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = matrix_i[i] % matrix_j[i];
}
////////////////////////////////////////////////////////////////////////////////

int main()
{
	////////////////////////////////////////////////////////////////////////////
	// Prepare the cpu data arrays

	// Fill the first array with the thread idx
	for (unsigned int i = 0; i < ARRAY_SIZE; ++i) {
		cpu_array_thread_numbers[i] = i;
	}
	
	// Fill the second array with random numbers
    // Random device
    std::random_device rnd_device;
    
	// Specify the engine and distribution
    std::mt19937 mersenne_engine {rnd_device()};
    
	// Uniform between 0 and 3
	std::uniform_int_distribution<unsigned int> dist {0, 3};
    
	// Create the lamba function
    auto gen = [&dist, &mersenne_engine]() {return dist(mersenne_engine);};

	// Use std generate to fill the random array
    std::generate(std::begin(cpu_array_random_numbers),
		std::end(cpu_array_random_numbers), gen);

	// Print
	std::cout << "[";
    for (auto i : cpu_array_thread_numbers) {
        std::cout << i << ",";
    }
	std::cout << "]\n";

	std::cout << "[";
    for (auto i : cpu_array_random_numbers) {
        std::cout << i << ",";
    }
	std::cout << "]\n";

	////////////////////////////////////////////////////////////////////////////
	// Prepare the GPU memory and execute the kernels

	// Declare pointers for GPU based params
	unsigned int * gpu_thread_number_array;
	unsigned int * gpu_thread_random_array;
	unsigned int * gpu_destination_array;

	// Get the CPU array to the GPU memory
	cudaMalloc((void **)&gpu_thread_number_array, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread_random_array, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_destination_array, ARRAY_SIZE_IN_BYTES);

	TIC();
	////////////////////////////////////////////////////////////////////////////
	// Execute addition
	cudaMemcpy(gpu_thread_number_array, cpu_array_thread_numbers.data(),
		ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread_random_array, cpu_array_random_numbers.data(),
		ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	add_matrices<<<N_BLOCKS, N_THREADS>>> (gpu_destination_array,
		gpu_thread_number_array, gpu_thread_random_array);

	// Retrieve the data from the GPU memory
	cudaMemcpy(results.data(), gpu_destination_array,
		ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	std::cout << "Results Addition: [";
    for (auto i : results) {
        std::cout << i << ",";
    }
	std::cout << "]\n";

	////////////////////////////////////////////////////////////////////////////
	// Execute subtraction
	cudaMemcpy(gpu_thread_number_array, cpu_array_thread_numbers.data(),
		ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread_random_array, cpu_array_random_numbers.data(),
		ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	subtract_matrices<<<N_BLOCKS, N_THREADS>>> (gpu_destination_array,
		gpu_thread_number_array, gpu_thread_random_array);

	// Retrieve the data from the GPU memory
	cudaMemcpy(results.data(), gpu_destination_array,
		ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	std::cout << "Results Subtraction: [";
    for (auto i : results) {
        std::cout << i << ",";
    }
	std::cout << "]\n";

	////////////////////////////////////////////////////////////////////////////
	// Execute multiplication
	cudaMemcpy(gpu_thread_number_array, cpu_array_thread_numbers.data(),
		ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread_random_array, cpu_array_random_numbers.data(),
		ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	multiply_matrices<<<N_BLOCKS, N_THREADS>>> (gpu_destination_array,
		gpu_thread_number_array, gpu_thread_random_array);

	// Retrieve the data from the GPU memory
	cudaMemcpy(results.data(), gpu_destination_array,
		ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	std::cout << "Results multiplication: [";
    for (auto i : results) {
        std::cout << i << ",";
    }
	std::cout << "]\n";

	////////////////////////////////////////////////////////////////////////////
	// Execute modulo division
	cudaMemcpy(gpu_thread_number_array, cpu_array_thread_numbers.data(),
		ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread_random_array, cpu_array_random_numbers.data(),
		ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	multiply_matrices<<<N_BLOCKS, N_THREADS>>> (gpu_destination_array,
		gpu_thread_number_array, gpu_thread_random_array);

	// Retrieve the data from the GPU memory
	cudaMemcpy(results.data(), gpu_destination_array,
		ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	std::cout << "Results Modulus: [";
    for (auto i : results) {
        std::cout << i << ",";
    }
	std::cout << "]\n";

	std::cout << "GPU took " << TOC<std::chrono::microseconds>() << " microseconds\n";
	////////////////////////////////////////////////////////////////////////////
	// Free the GPU memory
	cudaFree(gpu_thread_number_array);
	cudaFree(gpu_thread_random_array);
	cudaFree(gpu_destination_array);

	////////////////////////////////////////////////////////////////////////////
	// Print the results and stats


}
