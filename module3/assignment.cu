//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void warmUp()
{
}

/*
// computes multiplicity of arrays A,B and stores them in C
// input floating arrays A, B
// output array C
*/
__global__ void arrayCalculation(
	float* A,
	float* B,
	float* C)
{
	//printf("arrayCalculation\n");
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	C[thread_idx] = A[thread_idx] * B[thread_idx]* (thread_idx + 1);
}

/*
// computes multiplicity of arrays A,B and stores them in C
// conditional branching is applied depending on whether A[thread_idx]> B[thread_idx]
// input floating arrays A, B
// output array C
*/
__global__ void arrayCalculationBranch(
	float* A,
	float* B,
	float* C)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (A[thread_idx] > B[thread_idx]) {
		//printf("Condition 1 Hit\n");
		
		C[thread_idx] = A[thread_idx] * B[thread_idx] * (thread_idx + 1);
	}
	else {
		//printf("Condition 2 Hit\n");
		C[thread_idx] = (A[thread_idx] * B[thread_idx]) / (thread_idx + 1);
	}
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	// calculate number of blocks
	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
	
	unsigned int array_size_bytes = (sizeof(float) * (totalThreads));
	printf("Number of Threads elements: %d\n", totalThreads);
	printf("Block size: %d\n", blockSize);
	printf("Number of blocks: %d\n\n", numBlocks);
	// allocate memory
	float* cpu_array_A = (float*)malloc(array_size_bytes);
	float* cpu_array_B = (float*)malloc(array_size_bytes);
	float* cpu_array_C = (float*)malloc(array_size_bytes);
	float* cpu_array_C_Branch = (float*)malloc(array_size_bytes);

	// initialize input data
	printf("Initializing input...\n");
	for (int i = 0; i < totalThreads; ++i) {
		cpu_array_A[i] = (float)(i % 100) / 10.0f;
		cpu_array_B[i] = (float)((i + 50) % 100) / 10.0f;
	}


	// allocate GPU memory
	float* gpu_array_A;
	float* gpu_array_B;
	float* gpu_array_C;
	float* gpu_array_C_Branch;

	cudaMalloc((void **)&gpu_array_A, array_size_bytes);
	cudaMalloc((void **)&gpu_array_B, array_size_bytes);
	cudaMalloc((void **)&gpu_array_C, array_size_bytes);
	cudaMalloc((void **)&gpu_array_C_Branch, array_size_bytes);

	cudaMemcpy(gpu_array_A, cpu_array_A, array_size_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_array_B, cpu_array_B, array_size_bytes, cudaMemcpyHostToDevice);
	// Warm up GPU. Without this, the first time kernel call is made it increases run time
	warmUp<<<1, 1>>>();
	cudaDeviceSynchronize();

	// run GPU calculation without branching
	clock_t start = clock();

	arrayCalculation<<<numBlocks, blockSize>>>(
		gpu_array_A,
		gpu_array_B,
		gpu_array_C);

	cudaDeviceSynchronize();

	clock_t end = clock();

	double gpuTime_NoBranch =
		((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;


	// run GPU calculation with branching
	start = clock();

	arrayCalculationBranch<<<numBlocks, blockSize>>>(
		gpu_array_A,
		gpu_array_B,
		gpu_array_C_Branch);

	cudaDeviceSynchronize();

	end = clock();

	double gpuTime_Branching =
		((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;


	cudaMemcpy(cpu_array_C, gpu_array_C, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_array_C_Branch, gpu_array_C_Branch, array_size_bytes, cudaMemcpyDeviceToHost);

	// display result head
	printf("\nFirst 10 results:\n");

	for (int i = 0; i < 10 && i < totalThreads; ++i) {
		printf("cpu_array_C[%d] = %.2f    cpu_array_C_Branch[%d] = %.2f\n",
			i, cpu_array_C[i], i, cpu_array_C_Branch[i]);
	}

	// display performance results
	printf("\nGPU Performance\n");
	printf("-------------------------\n");
	printf("Without branching: %.4f ms\n", gpuTime_NoBranch);
	printf("With branching:    %.4f ms\n", gpuTime_Branching);
	// free memory
	cudaFree(gpu_array_A);
	cudaFree(gpu_array_B);
	cudaFree(gpu_array_C_Branch);
	cudaFree(gpu_array_C);
	
	free(cpu_array_A);
	free(cpu_array_B);
	free(cpu_array_C_Branch);
	free(cpu_array_C);

	return 0;
}