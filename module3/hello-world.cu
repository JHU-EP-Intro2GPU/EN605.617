// Modification of Ingemar Ragnemalm "Real Hello World!" program
// To compile execute below:
// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include <stdio.h>

#define BLOCK_SIZE 16




__global__ void hello(int * block)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = threadIdx.x;
}

void main_sub(unsigned int num_threads)
{

	unsigned int num_blocks = num_threads / BLOCK_SIZE;
	unsigned int array_size_bytes = sizeof(unsigned int) * num_threads;
	/* Declare  statically four arrays of ARRAY_SIZE each */
	unsigned int cpu_block[num_threads];
	/* Declare pointers for GPU based params */
	int *gpu_block;

	cudaMalloc((void **)&gpu_block, array_size_bytes);
	cudaMemcpy( gpu_block, cpu_block, array_size_bytes, cudaMemcpyHostToDevice );

	/* Execute our kernel */
	hello<<<num_blocks, BLOCK_SIZE>>>(gpu_block);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy( cpu_block, gpu_block, array_size_bytes, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < num_threads; i++)
	{
		printf("Calculated Thread: - Block: %2u\n",cpu_block[i]);
	}
}

int main()
{
	main_sub(16);
	main_sub(32);
	main_sub(48);
	main_sub(64);
	main_sub(79);
	return EXIT_SUCCESS;
}
