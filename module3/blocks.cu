#include <stdio.h>




__global__
void what_is_my_id(unsigned int * block, unsigned int * thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

void main_sub0(unsigned int thread_count, unsigned int block_size)
{
	unsigned int array_size_bytes = (sizeof(unsigned int) * (thread_count));
	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;
	/* Declare  two arrays of thread_count each */
	unsigned int cpu_block[thread_count];
	unsigned int cpu_thread[thread_count];
	cudaMalloc((void **)&gpu_block, array_size_bytes);
	cudaMalloc((void **)&gpu_thread, array_size_bytes);
	cudaMemcpy( cpu_block, gpu_block, array_size_bytes, cudaMemcpyHostToDevice );
	cudaMemcpy( cpu_thread, gpu_thread, array_size_bytes, cudaMemcpyHostToDevice );

	const unsigned int num_blocks = thread_count/block_size;
	const unsigned int threads_per_block = thread_count/num_blocks;

	/* Execute our kernel */
	what_is_my_id<<<num_blocks, threads_per_block>>>(gpu_block, gpu_thread);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy( cpu_block, gpu_block, array_size_bytes, cudaMemcpyDeviceToHost );
	cudaMemcpy( cpu_thread, gpu_thread, array_size_bytes, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block);
	cudaFree(gpu_thread);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < thread_count; i++)
	{
		printf("Thread: %2u - Block: %2u\n",cpu_thread[i],cpu_block[i]);
	}
}

int main()
{
	main_sub0(256, 16);
	main_sub0(512, 32);
	main_sub0(256, 8);
	main_sub0(512, 64);
	main_sub0(1024, 128);
	return EXIT_SUCCESS;
}
