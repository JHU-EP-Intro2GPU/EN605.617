#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <string>
#include "benchmarking.h"
#include "assignment.h"


// #define ARRAY_SIZE 64 // CPU array sizes
// #define N_THREADS 64 // Number of threads we want in parallel
// #define BLOCK_SIZE 64 // Threads per block
// #define n_blocks N_THREADS/BLOCK_SIZE // Calculate how many blocks will execute
// #define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE)) // for mallocs

int main(int argc, char * argv[])
{
	std::cout << "\nRUNNING ASSIGNMENT" << std::endl;
	////////////////////////////////////////////////////////////////////////////
	// Process command line args
	unsigned int block_size = 0; // Threads per block
	unsigned int n_threads = 0; // Total threads we want
	unsigned int n_blocks = 0; // Number of blocks to hold all the threads

	if (argc == 0) {
		std::cout << "Usage: ./file [block size] [number of threads per block]" << std::endl;
		return 0;
	}
	else {
		n_threads = std::stol(std::string(argv[1]));
		block_size = std::stol(std::string(argv[2]));
		n_blocks = n_threads / block_size;
	}

	std::vector<unsigned int> cpu_array_thread_numbers(n_threads);
	std::vector<unsigned int> cpu_array_random_numbers(n_threads);
	
	std::vector<unsigned int> results_addition(n_threads);
	std::vector<unsigned int> results_subtraction(n_threads);
	std::vector<unsigned int> results_multiplication(n_threads);
	std::vector<unsigned int> results_modulo(n_threads);

	////////////////////////////////////////////////////////////////////////////
	// Prepare the cpu data arrays

	// Fill the first array with the thread idx
	for (unsigned int i = 0; i < n_threads; ++i) {
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

	// Print test vectors
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
	cudaMalloc((void **)&gpu_thread_number_array,
		sizeof(unsigned int) * n_threads);
	cudaMalloc((void **)&gpu_thread_random_array,
		sizeof(unsigned int) * n_threads);
	cudaMalloc((void **)&gpu_destination_array,
		sizeof(unsigned int) * n_threads);

	TIC();
	////////////////////////////////////////////////////////////////////////////
	// Execute addition
	cudaMemcpy(gpu_thread_number_array, cpu_array_thread_numbers.data(),
		sizeof(unsigned int) * n_threads, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread_random_array, cpu_array_random_numbers.data(),
		sizeof(unsigned int) * n_threads, cudaMemcpyHostToDevice);

	add_matrices<<<n_blocks, block_size >>> (gpu_destination_array,
		gpu_thread_number_array, gpu_thread_random_array);

	// Retrieve the data from the GPU memory
	cudaMemcpy(results_addition.data(), gpu_destination_array,
		sizeof(unsigned int) * n_threads, cudaMemcpyDeviceToHost);

	////////////////////////////////////////////////////////////////////////////
	// Execute subtraction
	cudaMemcpy(gpu_thread_number_array, cpu_array_thread_numbers.data(),
		sizeof(unsigned int) * n_threads, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread_random_array, cpu_array_random_numbers.data(),
		sizeof(unsigned int) * n_threads, cudaMemcpyHostToDevice);

	subtract_matrices<<<n_blocks, block_size>>> (gpu_destination_array,
		gpu_thread_number_array, gpu_thread_random_array);

	// Retrieve the data from the GPU memory
	cudaMemcpy(results_subtraction.data(), gpu_destination_array,
		sizeof(unsigned int) * n_threads, cudaMemcpyDeviceToHost);

	////////////////////////////////////////////////////////////////////////////
	// Execute multiplication
	cudaMemcpy(gpu_thread_number_array, cpu_array_thread_numbers.data(),
		sizeof(unsigned int) * n_threads, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread_random_array, cpu_array_random_numbers.data(),
		sizeof(unsigned int) * n_threads, cudaMemcpyHostToDevice);

	multiply_matrices<<<n_blocks, block_size>>> (gpu_destination_array,
		gpu_thread_number_array, gpu_thread_random_array);

	// Retrieve the data from the GPU memory
	cudaMemcpy(results_multiplication.data(), gpu_destination_array,
		sizeof(unsigned int) * n_threads, cudaMemcpyDeviceToHost);

	////////////////////////////////////////////////////////////////////////////
	// Execute modulo division
	cudaMemcpy(gpu_thread_number_array, cpu_array_thread_numbers.data(),
		sizeof(unsigned int) * n_threads, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread_random_array, cpu_array_random_numbers.data(),
		sizeof(unsigned int) * n_threads, cudaMemcpyHostToDevice);

	modulo_matrices<<<n_blocks, block_size>>> (gpu_destination_array,
		gpu_thread_number_array, gpu_thread_random_array);

	// Retrieve the data from the GPU memory
	cudaMemcpy(results_modulo.data(), gpu_destination_array,
		sizeof(unsigned int) * n_threads, cudaMemcpyDeviceToHost);

	std::cout << "GPU took " << TOC<std::chrono::microseconds>() <<
	" microseconds to do all 4 operations (n_threads: " << n_threads
	<< ", n_blocks: " << n_blocks << ")\n";
	
	////////////////////////////////////////////////////////////////////////////
	// Print the results
	std::cout << "Results Addition:\n";
	print_vector(results_addition);
	std::cout << "Results Subtraction:\n";
	print_vector(results_subtraction);
	std::cout << "Results multiplication:\n";
    print_vector(results_multiplication);
	std::cout << "Results Modulus:\n";
    print_vector(results_modulo);

	////////////////////////////////////////////////////////////////////////////
	// Free the GPU memory
	cudaFree(gpu_thread_number_array);
	cudaFree(gpu_thread_random_array);
	cudaFree(gpu_destination_array);

	return 0;
}
