// Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_ELEMENTS 1000000

__global__ void addArrays(const int *a, const int *b, int *c, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (i < n)
	{
		c[i] = a[i] + b[i];
		i += stride;
	}
}

void addArraysCPU(const int *a, const int *b, int *c, int n)
{
	for (int i = 0; i < n; i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char **argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	int numElements = NUM_ELEMENTS;

	if (argc >= 2)
	{
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3)
	{
		blockSize = atoi(argv[2]);
	}
	if (argc >= 4)
	{
		numElements = atoi(argv[3]);
	}

	int numBlocks = totalThreads / blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0)
	{
		++numBlocks;
		totalThreads = numBlocks * blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	size_t sz = sizeof(int) * numElements;

	int *a = (int *)malloc(sz);
	int *b = (int *)malloc(sz);
	int *c = (int *)malloc(sz);

	for (int i = 0; i < numElements; i++)
	{
		a[i] = i;
		b[i] = i;
	}

	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, sz);
	cudaMalloc(&d_b, sz);
	cudaMalloc(&d_c, sz);

	cudaMemcpy(d_a, a, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sz, cudaMemcpyHostToDevice);

	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	cudaEventRecord(startEvent);
	addArrays<<<numBlocks, blockSize>>>(d_a, d_b, d_c, numElements);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);

	float gpuTime;
	cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);

	cudaMemcpy(c, d_c, sz, cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// do something
	clock_t startTime = clock();
	addArraysCPU(a, b, c, numElements);
	double cpuTime = (double)(clock() - startTime) / CLOCKS_PER_SEC * 1000.0;

	printf("CPU time: %.3f ms\n", cpuTime);
	printf("CUDA time: %.3f ms\n", gpuTime);

	for (int i = 0; i < 100; i++)
	{
		// printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	free(a);
	free(b);
	free(c);
}
