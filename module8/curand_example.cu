#include <unistd.h>
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define MAX 100
#define N 25
#define M 20

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              thread_idx, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[thread_idx]);
}
 
/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, unsigned int* numbers) {
  /* curand works like rand - except that it takes a state as a parameter */
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  numbers[thread_idx] = curand(&states[thread_idx]) % MAX;
}
 
/* this GPU kernel function calculates a random number and stores it in the parameter */
__global__ void random0(int* result) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* we have to initialize the state */
  curand_init(1, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  /* curand works like rand - except that it takes a state as a parameter */
  *result = curand(&state) % MAX;
}
 
/* this GPU kernel function calculates a random number and stores it in the parameter */
__global__ void random1(unsigned int seed, int* result) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* we have to initialize the state */
  curand_init(seed, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  /* curand works like rand - except that it takes a state as a parameter */
  *result = curand(&state) % MAX;
}

int main( ) {
  /* allocate an int on the GPU */
  int* gpu_x0;
  cudaMalloc((void**) &gpu_x0, sizeof(int));

  /* invoke the GPU to initialize all of the random states */
  random0<<<1, 1>>>(gpu_x0);

  /* copy the random number back */
  int x0;
  cudaMemcpy(&x0, gpu_x0, sizeof(int), cudaMemcpyDeviceToHost);

  printf("Random number = %d.\n", x0);

  /* free the memory we allocated */
  cudaFree(gpu_x0);

   /* allocate an int on the GPU */
  int* gpu_x1;
  cudaMalloc((void**) &gpu_x1, sizeof(int));

  /* invoke the GPU to initialize all of the random states */
  random1<<<1, 1>>>(time(NULL), gpu_x1);

  /* copy the random number back */
  int x1;
  cudaMemcpy(&x1, gpu_x1, sizeof(int), cudaMemcpyDeviceToHost);

  printf("Random number = %d.\n", x1);

  /* free the memory we allocated */
  cudaFree(gpu_x1);
 
   /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t* states;

  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states, N * M * sizeof(curandState_t));

  /* invoke the GPU to initialize all of the random states */
  init<<<N, M>>>(time(0), states);

  /* allocate an array of unsigned ints on the CPU and GPU */
  unsigned int cpu_nums[N * M];
  unsigned int* gpu_nums;
  cudaMalloc((void**) &gpu_nums, N * M * sizeof(unsigned int));

  /* invoke the kernel to get some random numbers */
  randoms<<<N, M>>>(states, gpu_nums);

  /* copy the random numbers back */
  cudaMemcpy(cpu_nums, gpu_nums, N * M * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  /* print them out */
  for (int i = 0; i < N; i++) {
      for (int j =0; j < M; j++){
          printf("%u ", cpu_nums[i * M + j]);
      }
      printf("\n");
  }

  /* free the memory we allocated for the states and numbers */
  cudaFree(states);
  cudaFree(gpu_nums);
 
  return 0;
}