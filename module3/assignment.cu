// Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <chrono>

#define NUM_ELEMENTS 10000000

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

__global__ void addArraysWithBranching(const int *a, const int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (i < n)
    {
        if (i % 2 == 0)
        {
            c[i] = a[i] + b[i];
        }
        else
        {
            c[i] = a[i] - b[i];
        }
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

void addArraysCPUWithBranching(const int *a, const int *b, int *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (i % 2 == 0)
        {
            c[i] = a[i] + b[i];
        }
        else
        {
            c[i] = a[i] - b[i];
        }
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

        printf("Warning: Total thread count is not evenly divisible by "
               "the block "
               "size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    size_t sz = sizeof(int) * numElements;

    // Initialize arrays
    int *a = (int *)malloc(sz);
    int *b = (int *)malloc(sz);
    int *c = (int *)malloc(sz);

    for (int i = 0; i < numElements; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    // Process with GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sz);
    cudaMalloc(&d_b, sz);
    cudaMalloc(&d_c, sz);

    cudaMemcpy(d_a, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sz, cudaMemcpyHostToDevice);
    addArrays<<<numBlocks, blockSize>>>(d_a, d_b, d_c, numElements);
    auto startTime = std::chrono::high_resolution_clock::now();
    addArrays<<<numBlocks, blockSize>>>(d_a, d_b, d_c, numElements);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();

    cudaMemcpy(c, d_c, sz, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Process with CPU
    startTime = std::chrono::high_resolution_clock::now();
    addArraysCPU(a, b, c, numElements);
    endTime = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();

    printf("CPU time (without branching): %ld ns\n", cpuTime);
    printf("CUDA time (without branching): %ld ns\n", gpuTime);

    // Process branching version with GPU

    cudaMalloc(&d_a, sz);
    cudaMalloc(&d_b, sz);
    cudaMalloc(&d_c, sz);

    cudaMemcpy(d_a, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sz, cudaMemcpyHostToDevice);
    startTime = std::chrono::high_resolution_clock::now();
    addArraysWithBranching<<<numBlocks, blockSize>>>(d_a, d_b, d_c, numElements);
    endTime = std::chrono::high_resolution_clock::now();
    auto gpuTimeWithBranching = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();

    cudaMemcpy(c, d_c, sz, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Process branching version with CPU
    startTime = std::chrono::high_resolution_clock::now();
    addArraysCPUWithBranching(a, b, c, numElements);
    endTime = std::chrono::high_resolution_clock::now();
    auto cpuTimeWithBranching = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();

    printf("CPU time (with branching): %ld ns\n", cpuTimeWithBranching);
    printf("CUDA time (with branching): %ld ns\n", gpuTimeWithBranching);
    free(a);
    free(b);
    free(c);
}
