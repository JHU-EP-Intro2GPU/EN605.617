#include <stdio.h>

#define ARRAY_SIZE 64  /* grand total number of threads to be executed, not necessailry in parallel*/

#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

/* Declare  statically two arrays of ARRAY_SIZE each */
unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_result[ARRAY_SIZE];

__global__
void identity_and_thread_block_summation(unsigned int * block, unsigned int * thread, unsigned int * result)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;

	result[thread_idx] = block[thread_idx] + thread[thread_idx];
}

void main_sub0()
{

	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;
	unsigned int *gpu_result;

	cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_result, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy( cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	const unsigned int num_blocks = ARRAY_SIZE/64;
	const unsigned int num_threads = ARRAY_SIZE/num_blocks;

	/* Execute our kernel */
	identity_and_thread_block_summation<<<num_blocks, num_threads>>>(gpu_block, gpu_thread, gpu_result);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy( cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block);
	cudaFree(gpu_thread);
	cudaFree(gpu_result);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("Thread: %2u - Block: %2u - Thread+Block: %2u\n",cpu_thread[i],cpu_block[i],cpu_result[i]);
	}
}

int main()
{
	main_sub0();

	return EXIT_SUCCESS;
}
