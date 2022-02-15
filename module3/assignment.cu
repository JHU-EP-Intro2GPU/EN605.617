#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include "assignment.h"
#include "benchmarking.h"

template<typename T>
void print_vector(const std::vector<T> & vec) {
	std::cout << "[";
    for (auto i : vec) {
        std::cout << i << ",";
    }
	std::cout << "]\n";
}

#define ARRAY_SIZE 64 // CPU array sizes

#define N_THREADS 64 // Number of threads we want in parallel
#define BLOCK_SIZE 64 // Threads per block
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

int main(int argc, char * argv[])
{
	////////////////////////////////////////////////////////////////////////////
	if (argc == 0) {
		std::cout << "Usage: ./file {block size} {number of threads per block}" << std::endl;
		
	}
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
	std::cout << "Thread number vector:\n";
	print_vector(cpu_array_thread_numbers);
	std::cout << "Random number vector:\n";
	print_vector(cpu_array_random_numbers);

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

	std::cout << "Results Addition:\n";
	print_vector(results);

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

	std::cout << "Results Subtraction:\n";
	print_vector(results);

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

	std::cout << "Results multiplication:\n";
    print_vector(results);

	////////////////////////////////////////////////////////////////////////////
	// Execute modulo division
	cudaMemcpy(gpu_thread_number_array, cpu_array_thread_numbers.data(),
		ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread_random_array, cpu_array_random_numbers.data(),
		ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	modulo_matrices<<<N_BLOCKS, N_THREADS>>> (gpu_destination_array,
		gpu_thread_number_array, gpu_thread_random_array);

	// Retrieve the data from the GPU memory
	cudaMemcpy(results.data(), gpu_destination_array,
		ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	std::cout << "Results Modulus:\n";
    print_vector(results);

	std::cout << "GPU took " << TOC<std::chrono::microseconds>() << " microseconds\n";
	////////////////////////////////////////////////////////////////////////////
	// Free the GPU memory
	cudaFree(gpu_thread_number_array);
	cudaFree(gpu_thread_random_array);
	cudaFree(gpu_destination_array);

	////////////////////////////////////////////////////////////////////////////
	// Print the results and stats


}
