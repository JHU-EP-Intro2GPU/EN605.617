#include <stdio.h>
#include <stdlib.h>

/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
*/

//Kernel that adds two vectors
__global__
void add_ab(int *a, const int *b)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	a[thread_idx] += b[thread_idx];
}

// Helper function to generate a random number within a defined range
int random(int min, int max){
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

void run_vector_add(int * num_threads, int * block_size)
{ 
    printf("Running random vector add with %d threads and a block size of %d\n",*num_threads,*num_threads);
    int array_size = *num_threads;
    int array_size_in_bytes = (sizeof(int) * (array_size));

    /* Randomly generate input vectors and dynamically allocate their memory */
    int * a; 
    int * b;
    
    a = (int*)malloc(array_size * sizeof(int));
    b = (int*)malloc(array_size * sizeof(int));

    int i;
    for (i = 0; i < array_size; i++) {
        a[i] = random(0,100);
    }
    for (i = 0; i < array_size; i++) {
        b[i] = random(0,100);
    }

	/* Declare pointers for GPU based params */
    int *a_d;
	int *b_d;

	cudaMalloc((void**)&a_d, array_size_in_bytes);
	cudaMalloc((void**)&b_d, array_size_in_bytes);
	cudaMemcpy( a_d, a, array_size_in_bytes, cudaMemcpyHostToDevice );
	cudaMemcpy( b_d, b, array_size_in_bytes, cudaMemcpyHostToDevice );

	const unsigned int num_blocks = array_size / *block_size;
	const unsigned int num_threads_per_blk = array_size/num_blocks;

	/* Execute our kernel */
	add_ab<<<num_blocks, num_threads_per_blk>>>(a_d, b_d);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy(a, a_d, array_size_in_bytes, cudaMemcpyDeviceToHost );
	cudaFree(a_d);
	cudaFree(b_d);

	/* Iterate through the result array and print */
    for(unsigned int i = 0; i < array_size; i++)
	{
		printf("Sum #%d: %d\n",i,a[i]);
	}
}

int main()
{   
    int num_threads;
    int block_size;

    printf("\nExample Addition 0\n\n");

    num_threads = 256;
    block_size = 16;

    run_vector_add(&num_threads, &block_size);

    printf("\nExample Addition 1: Changing number of threads\n\n");

    num_threads = 512;
    block_size = 16;

    run_vector_add(&num_threads, &block_size);

    printf("\nExample Addition 2: Changing number of threads\n\n");

    num_threads = 1024;
    block_size = 16;

    run_vector_add(&num_threads, &block_size);

    printf("\nExample Addition 3: Changing number of block size\n\n");

    num_threads = 256;
    block_size = 32;

    run_vector_add(&num_threads, &block_size);

    printf("\nExample Addition 4: Changing number of block size\n\n");

    num_threads = 256;
    block_size = 64;

    run_vector_add(&num_threads, &block_size);
    
    printf("\nCustom Addition: Allow for user input\n\n");

    printf("Enter total number of threads: ");
    scanf("%d", &num_threads);
 
    printf("Enter total threads per block (block size): ");
    scanf("%d", &block_size);

    run_vector_add(&num_threads, &block_size);

	return EXIT_SUCCESS;
}
