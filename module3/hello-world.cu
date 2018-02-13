// Modification of Ingemar Ragnemalm "Real Hello World!" program
// To compile execute below:
// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include <stdio.h>

#define N 16
#define BLOCK_SIZE 16
#define NUM_BLOCKS N/BLOCK_SIZE

#define ARRAY_SIZE N
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

/* Declare  statically four arrays of ARRAY_SIZE each */
unsigned int cpu_block[ARRAY_SIZE];

__global__ 
void hello(int * block)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = threadIdx.x;
}

void main_sub()
{

	/* Declare pointers for GPU based params */
	int *gpu_block;

	cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy( gpu_block, cpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	/* Execute our kernel */
	hello<<<NUM_BLOCKS, BLOCK_SIZE>>>(gpu_block);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy( cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("Calculated Thread: - Block: %2u\n",cpu_block[i]);
	}
}

int main()
{
	main_sub();

	return EXIT_SUCCESS;
}
