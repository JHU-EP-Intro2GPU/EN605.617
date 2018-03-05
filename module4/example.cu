// Intro to GPU Programming/
/ Module 7 Assignment
//
//
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <math.h>

static const uint32_t DEFAULT_NUM_BLOCKS = 2048;
....
static const int DEFAULT_SEED = -1;
static float MIN_VAL = -10000.0;

// Print usage information
static void usage(){    
    printf("Usage: ./assignment7 [-t <num_threads>] [-b <num_blocks>] [-s 
 
  ] [-i 
  
   ] [-p] [-h]\n");
   
 printf("\t-t: Specify the number of threads. <num_threads> must be greater than 0. Optional (default %u)\n", DEFAULT_NUM_THREADS);
   
    printf("\t-b: Specify the number of blocks. <num_blocks> must be greater than 0. Optional (default %u)\n", DEFAULT_NUM_BLOCKS);
   
    printf("\t-s: Specify a seed for the random number generator. 
   
     must be greater or equal to zero. Optional (default is random)\n");
    
    printf("\t-i: Specify the number of kernel iterations for benchmarking. Optional (default is %u)\n", DEFAULT_NUM_ITERATIONS);
    
 ....
    
}
   
  
 

// Structure that holds program arguments specifying number of threads/blocks
// to use.
typedef struct {    
    uint32_t num_threads;
    uint32_t num_blocks;
    int seed;
    ....
} Arguments;

static void check_arg(const int value, const int c){    
     if (value <= 0) {        
         printf("Error: invalid value (%d) for arg (%c). Must be positive\n", value, c);
        exit(-1);
    }
}

    // Parse the command line arguments using getopt and return an Argument structure
    // GetOpt requies the POSIX C Library
static Arguments parse_arguments(const int argc, char ** argv){   
    // Argument format string for getopt    static const char * _ARG_STR = "ht:b:s:i:";
    // Initialize arguments to their default values    
    Arguments args;    
    args.num_threads = DEFAULT_NUM_THREADS;    
    args.num_blocks = DEFAULT_NUM_BLOCKS;
    args.seed = DEFAULT_SEED;    args.iterations = DEFAULT_NUM_ITERATIONS;        
    // Parse any command line options
    int c;
    int value;
    while ((c = getopt(argc, argv, _ARG_STR)) != -1) {
        switch (c) {
            case 't':
                value = atoi(optarg);
                ...
                break;
            case 'b':
                // Normal argument
                value = atoi(optarg);
                check_arg(value, c);
                ...
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

static float rand_float(){
    const float scale = rand() / (float) RAND_MAX;
    return MIN_VAL + scale * (MAX_VAL - MIN_VAL);
}

static float * alloc_vector(const uint32_t len, const int pinned){
    void * vec;
    if (pinned) {
        ....
    } else {
        vec = malloc(len * sizeof(float));
    }
    return (float *) vec;
}

static void free_vector(float * vec, int pinned){
    if (pinned) {
        cudaFreeHost(vec);
    } else {
        free(vec);
    }
}

// Initialize two random vector of floats in the range (-100, 100) of length len.
// If seed > 0, use it to seed the random number generator.
static void init_random_vectors(float ** a, float ** result, const int seed, const uint32_t len, const int pinned){
    if (seed >= 0) {
        srand(seed);
    }
    float * vecA = alloc_vector(len, pinned);
    assert(vecA != NULL);
    float * vecResult = alloc_vector(len, pinned);
    assert(vecResult != NULL);
    for (uint32_t i = 0; i < len; i++) {
        ...
    }
    *a = vecA;
    *result = vecResult;
}

// calculate square root of each element using fast-inverse square root method
__global__ void fast_sqrt(const float * a, float * output){
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    float in = a[idx];
    int32_t i;
    float x2, y;
    const float threehalves = 1.5f;
    x2 = in * 0.5;
    y = in;
    i = *((int32_t *) &in);
    i = 0x5f3759df - (i >> 1);
    y = *((float *) &i);
    y = y * (threehalves - (x2 * y * y));
    y = y * (threehalves - (x2 * y * y));
    output[idx] = 1.0 / y;
}

// Make sure results match within some error (1%)
static void check_result(const float * const a, const float * const result, const uint32_t len){
   for (uint32_t i = 0; i < len; i++) { 
       const float cpu_result = sqrt(a[i]);
       const float epsilon = fabs(cpu_result * .01);
       if (fabs(cpu_result - result[i]) < 0.0001 ) { 
         // special case when results are very, very small
          continue;
        }
         if (fabs(cpu_result - result[i]) > epsilon) {
            printf("Error: CPU and GPU results do not agree at index %u (cpu = %f, gpu = %f)\n", i, cpu_result, result[i]);
            return;
        }
    }
}

// Allocate cuda memory, generate the random vectors, and run the cuda kernel
static void run_cuda(Arguments args){
    float elapsed_s = 0.0;
    float milliseconds = 0.0;
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t start, stop;
    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Create streams 
    for (uint8_t snum = 0; snum < NUM_STREAMS; snum++) {
        cudaStreamCreate(streams + snum);
    }
    for (uint32_t i = 0; i < args.iterations; i++) {
       // Allocate Host memory
        float * a = NULL;
        float * result;
        init_random_vectors(&a, &result, args.seed, NUM_STREAMS * args.num_threads, args.pinned);
        // Allocate GPU Memory
        float * gpu_a = NULL;
        float * gpu_result = NULL;
        uint32_t array_len_bytes = NUM_STREAMS * args.num_threads * sizeof(float);
        cudaMalloc((void **) &gpu_a, array_len_bytes);
        cudaMalloc((void **) &gpu_result, array_len_bytes);
        // Start execution time measurement
        cudaEventRecord(start);
        const uint32_t threads_per_block = args.num_threads / args.num_blocks;
        if (args.depth_first_exeuction) {
            for (uint8_t snum = 0; snum < NUM_STREAMS; snum++) {
                ...
            }
        } else {
            for (uint8_t snum = 0; snum < NUM_STREAMS; snum++) {
                ...
            }
            for (uint8_t snum = 0; snum < NUM_STREAMS; snum++) {
                const int offset = snum * args.num_threads;
                fast_sqrt<<<args.num_blocks, threads_per_block, 0, streams[snum]>>>(gpu_a + offset, gpu_result + offset);
            }
            for (uint8_t snum = 0; snum < NUM_STREAMS; snum++) {
                ...
            }
        }
        cudaEventRecord(stop);
        // Calculate execution time
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
       ...
        printf("\nRan %u iterations\n", args.iterations);    
        printf("Average execution time of %f seconds while using %s-first execution\n", avg_elapsed,            args.depth_first_exeuction? "depth" : "breadth");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        for (uint8_t snum = 0; snum < NUM_STREAMS; snum++) {
          cudaStreamDestroy(streams[snum]);
        }
}

int main(int argc, char ** argv)
{
    Arguments args = parse_arguments(argc, argv);
    printf("Num Threads: %u, Num Blocks: %u\n", args.num_threads, args.num_blocks);
    // always use pinned memory for streams
    ...
    run_cuda(args);
     return EXIT_SUCCESS;
}