//Based on the work of Andrew Krepps
#include <stdio.h>
#include <vector>
#include <time.h>
#include <cstdlib>

#define N (1 << 24) // array size

__global__
void add(const int *A, const int *B, int *result, int arrSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < arrSize; i += stride) {
        result[i] = A[i] + B[i];
    }
}

__global__
void add_w_branch(const int *A, const int *B, int *result, int arrSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < arrSize; i += stride) {
		if (A[i] % 2 == 0) {
			result[i] = A[i] + B[i];
		} 
		else {
			result[i] = A[i] - B[i];
		}
    }
}

void do_work(int blockSize, int numBlocks, bool useBranching) {
	const int array_size_bytes = sizeof(int) * N;
	srand(time(NULL));

	// populate arrays A, B
	int *a_cpu = (int*)malloc(array_size_bytes);
	int *b_cpu = (int*)malloc(array_size_bytes);
	int *res_cpu = (int*)malloc(array_size_bytes);
	for (int i=0; i<N; i++) {
		a_cpu[i] = rand();
		b_cpu[i] = rand();
	}

	// set up GPU
	int *a_gpu, *b_gpu, *res_gpu;

	cudaMalloc((void **)&a_gpu, array_size_bytes);
	cudaMalloc((void **)&b_gpu, array_size_bytes);
	cudaMalloc((void **)&res_gpu, array_size_bytes);
	cudaMemcpy(a_gpu, a_cpu, array_size_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b_cpu, array_size_bytes, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start);


	// execute kernel
	if (!useBranching) {
		add<<<numBlocks, blockSize>>>(a_gpu, b_gpu, res_gpu, N);
	} else {
		add_w_branch<<<numBlocks, blockSize>>>(a_gpu, b_gpu, res_gpu, N);
	}
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
	float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);

	// free arrays on gpu
	cudaEventDestroy(start);
    cudaEventDestroy(stop);
	cudaMemcpy(res_cpu, res_gpu, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaFree(a_gpu);
	cudaFree(b_gpu);
	cudaFree(res_gpu);

	printf("\nKernel execution time (%s): %.3f ms\n", (useBranching ? "branching" : "non-branching"), time_ms);

}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	bool useBranching = false;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}
	if (argc >= 4) {
		useBranching = (atoi(argv[3]) != 0); 
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	printf("run gpu version (%s) with totalThreads %d, blockSize %d, numBlocks %d\n", (useBranching ? "branching" : "non-branching"), totalThreads, blockSize, numBlocks);
	do_work(blockSize, numBlocks, useBranching);

}
