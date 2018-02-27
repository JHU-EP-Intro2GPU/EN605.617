#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <math.h>

/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
Assignment: Module 4
Resources: https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
*/
static const uint32_t DEFAULT_NUM_THREADS = 1024;
static const uint32_t DEFAULT_BLOCK_SIZE = 16;

static void usage(){    
    printf("Usage: ./assignment4 [-t <num_threads>] [-b <block_size>] [-h]\n");
   
    printf("\t-t: Specify the number of threads. <num_threads> must be greater than 0. Optional (default %u)\n", DEFAULT_NUM_THREADS);
   
    printf("\t-b: Specify the size of each block. <block_size> must be greater than 0. Optional (default %u)\n", DEFAULT_BLOCK_SIZE);    
}
    

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

void measure_kern_speed(Arguments args, int * a , int * b, int * a_d, int * b_d)
{ 
    // create events for timing
    cudaEvent_t startEvent, stopEvent; 
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    int array_size = args.num_threads;
    const unsigned int array_size_in_bytes = array_size * sizeof(int);


    cudaEventRecord(startEvent, 0);
	cudaMemcpy( a_d, a, array_size_in_bytes, cudaMemcpyHostToDevice );
	cudaMemcpy( b_d, b, array_size_in_bytes, cudaMemcpyHostToDevice );
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

	const unsigned int num_blocks = array_size / args.block_size;
	const unsigned int num_threads_per_blk = array_size/num_blocks;

    float time;
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("  Host to Device bandwidth (GB/s): %f\n", array_size_in_bytes * 1e-6  * 2 / time);

	/* Execute our kernel */
	add_ab<<<num_blocks, num_threads_per_blk>>>(a_d, b_d);

    cudaEventRecord(startEvent, 0);
	cudaMemcpy(a, a_d, array_size_in_bytes, cudaMemcpyDeviceToHost );
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    printf("  Device to Host bandwidth (GB/s): %f\n", array_size_in_bytes * 1e-6 / time);
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void run_experemnt(Arguments args){
    int * a; 
    int * b;
    int *h_aPinned, *h_bPinned;

    int array_size = args.num_threads;
    const unsigned int array_size_in_bytes = array_size * sizeof(int);

    /* Randomly generate input vectors and dynamically allocate their memory */
    a = (int*)malloc(array_size_in_bytes);
    b = (int*)malloc(array_size_in_bytes);

    cudaMallocHost((void**)&h_aPinned, array_size_in_bytes);
    cudaMallocHost((void**)&h_bPinned, array_size_in_bytes);

    int i;
    for (i = 0; i < array_size; i++) {
        a[i] = random(0,100);
    }
    for (i = 0; i < array_size; i++) {
        b[i] = random(0,100);
    }
    memcpy(h_aPinned, a, array_size_in_bytes);
    memcpy(h_bPinned, b, array_size_in_bytes);

    /* Declare pointers for GPU based params */
    int *a_d;
    int *b_d;

    cudaMalloc((void**)&a_d, array_size_in_bytes);
    cudaMalloc((void**)&b_d, array_size_in_bytes);

    printf("Simple Pagaeable\n");
    measure_kern_speed(args, a, b, a_d, b_d);
    printf("Simple Pinned\n");
    measure_kern_speed(args, h_aPinned, h_bPinned, a_d, b_d);

    //free memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(a);
    free(b);
}

int main(int argc, char ** argv)
{
    Arguments args = parse_arguments(argc, argv);
    printf("Num Threads: %u, Block Size: %u\n", args.num_threads, args.block_size);

    run_experemnt(args);
    
	return EXIT_SUCCESS;
}
