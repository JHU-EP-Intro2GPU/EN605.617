#include <stdio.h>
#include <stdlib.h>

/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
*/

// Structure that holds program arguments specifying number of threads/blocks
// to use.
typedef struct {    
    uint32_t num_threads;
    uint32_t block_size;
} Arguments;

// Parse the command line arguments using getopt and return an Argument structure
// GetOpt requies the POSIX C Library
static Arguments parse_arguments(const int argc, char ** argv){   
    // Argument format string for getopt
    static const char * _ARG_STR = "ht:b:";
    // Initialize arguments to their default values    
    Arguments args;    
    args.num_threads = DEFAULT_NUM_THREADS;    
    args.block_size = DEFAULT_BLOCK_SIZE;
    // Parse any command line options
    int c;
    int value;
    while ((c = getopt(argc, argv, _ARG_STR)) != -1) {
        switch (c) {
            case 't':
                value = atoi(optarg);
                args.num_threads = value;
                break;
            case 'b':
                // Normal argument
                value = atoi(optarg);
                args.block_size = value;
                break;
            case 'h':
                // 'help': print usage, then exit
                // note the fall through
                usage();
            default:
                exit(-1);
        }
    }
    return args;
}

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

int main(int argc, char ** argv)
{
    Arguments args = parse_arguments(argc, argv);
    printf("Num Threads: %u, Block Size: %u\n", args.num_threads, args.block_size);

    run_vector_add(args.num_threads, args.block_size);
    
	return EXIT_SUCCESS;
}