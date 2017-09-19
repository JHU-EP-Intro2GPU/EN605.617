// EN.605.417
// Bill Burgett
// Assignment for module 3

#include <stdio.h>
#include <iostream>
using namespace std;

/*#include "cuPrintf.cuh"
#include "cuPrintf.cu"*/

#define N 16
#define BLOCK_SIZE 16
#define NUM_BLOCKS N/BLOCK_SIZE

#define ARRAY_SIZE 1024
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

//#define ARRAY_SIZE N
//#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

/* Declare  statically four arrays of ARRAY_SIZE each */
unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_result[ARRAY_SIZE];
//int inputA[ARRAY_SIZE];
//int inputB[ARRAY_SIZE];

__global__ 
void doTheThing(unsigned int * block, unsigned int * thread, unsigned int * result, unsigned int * inputA, unsigned int * inputB)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
	result[thread_idx] = inputA[thread_idx] + inputB[thread_idx];
	//printf("thread_idx %d - blockIdx.x %d - threadIdx.x %d - inputA[i] %d - inputB[i] %d\n", thread_idx, blockIdx.x, threadIdx.x, inputA[thread_idx], inputB[thread_idx]);
}

void main_sub()
{
	/* initialize pointer arrays*/
	/*int numOfThreads, blockSize;
	cout << "Enter number of threads (between 1 and 99): ";
	cin >> numOfThreads;
	cout << "";
	if (!(numOfThreads > 0) && (numOfThreads < 100)) {
		cout << "Invalid input detected, defaulting to 64 threads";
		cout << "";
	}

	cout << "Enter blocksize (as a factor of 4 between 1 and 4): ";
	cin >> blockSize;
	if (!(blockSize > 0) && (blockSize < 4)) {
		cout << "Invalid input detected, defaulting to block size = 1";
		cout << "";
	}*/

	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;
	unsigned int *gpu_result;

	//int blockSize = 1;
	//int threadCount = 64;

	cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_result, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_block, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_thread, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	const unsigned int num_blocks = 1;// ARRAY_SIZE / 16;
	const unsigned int num_threads = ARRAY_SIZE / num_blocks;

	unsigned int *gpu_inputA, *gpu_inputB;
	unsigned int *inputA, *inputB;
	inputA = (unsigned int *)malloc(ARRAY_SIZE * sizeof(unsigned int));
	inputB = (unsigned int *)malloc(ARRAY_SIZE * sizeof(unsigned int));
	for (unsigned int i = 0; i < ARRAY_SIZE; i++) {
		inputA[i] = i;
		inputB[i] = i * 2;
		//printf("A %i and B %i\n", inputA[i], inputB[i]);
	}
	cudaMalloc((void **)&gpu_inputA, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_inputB, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(gpu_inputA, inputA, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_inputB, inputB, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	/* Execute our kernel */
	doTheThing << <num_blocks, num_threads >> >(gpu_block, gpu_thread, gpu_result, gpu_inputA, gpu_inputB);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_inputA);
	cudaFree(gpu_inputB);
	cudaFree(gpu_result);
	cudaFree(gpu_block);
	cudaFree(gpu_thread);

	/* Iterate through the arrays and print */
	for (unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("Thread: %2u - Block: %2u. Result is: %2u\n", cpu_thread[i], cpu_block[i], cpu_result[i]);
	}
}

void main_sub1()
{
	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;
	unsigned int *gpu_result;

	cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_result, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_block, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_thread, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	const unsigned int num_blocks = 16;
	const unsigned int num_threads = ARRAY_SIZE / num_blocks;

	unsigned int *gpu_inputA, *gpu_inputB;
	unsigned int *inputA, *inputB;
	inputA = (unsigned int *)malloc(ARRAY_SIZE * sizeof(unsigned int));
	inputB = (unsigned int *)malloc(ARRAY_SIZE * sizeof(unsigned int));
	for (unsigned int i = 0; i < ARRAY_SIZE; i++) {
		inputA[i] = i;
		inputB[i] = i * 2;
	}
	cudaMalloc((void **)&gpu_inputA, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_inputB, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(gpu_inputA, inputA, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_inputB, inputB, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	/* Execute our kernel */
	doTheThing << <num_blocks, num_threads >> >(gpu_block, gpu_thread, gpu_result, gpu_inputA, gpu_inputB);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_inputA);
	cudaFree(gpu_inputB);
	cudaFree(gpu_result);
	cudaFree(gpu_block);
	cudaFree(gpu_thread);

	/* Iterate through the arrays and print */
	for (unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("Thread: %2u - Block: %2u. Result is: %2u\n", cpu_thread[i], cpu_block[i], cpu_result[i]);
	}
}

void main_sub2()
{
	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;
	unsigned int *gpu_result;

	cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_result, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_block, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_thread, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	const unsigned int num_blocks = 8;
	const unsigned int num_threads = ARRAY_SIZE / num_blocks;

	unsigned int *gpu_inputA, *gpu_inputB;
	unsigned int *inputA, *inputB;
	inputA = (unsigned int *)malloc(ARRAY_SIZE * sizeof(unsigned int));
	inputB = (unsigned int *)malloc(ARRAY_SIZE * sizeof(unsigned int));
	for (unsigned int i = 0; i < ARRAY_SIZE; i++) {
		inputA[i] = i;
		inputB[i] = i * 2;
	}
	cudaMalloc((void **)&gpu_inputA, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_inputB, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(gpu_inputA, inputA, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_inputB, inputB, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	/* Execute our kernel */
	doTheThing << <num_blocks, num_threads >> >(gpu_block, gpu_thread, gpu_result, gpu_inputA, gpu_inputB);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_inputA);
	cudaFree(gpu_inputB);
	cudaFree(gpu_result);
	cudaFree(gpu_block);
	cudaFree(gpu_thread);

	/* Iterate through the arrays and print */
	for (unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("Thread: %2u - Block: %2u. Result is: %2u\n", cpu_thread[i], cpu_block[i], cpu_result[i]);
	}
}

int main()
{
	main_sub();
	main_sub1();
	main_sub2();

	return EXIT_SUCCESS;
}
