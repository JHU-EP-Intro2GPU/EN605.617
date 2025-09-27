//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <cstdlib>

#define N (1 << 24) // array size

void add(int* A, int* B, int* result, int index) {
	result[index] = A[index] + B[index];
}

void add_w_branch(int* A, int* B, int* result, int index) {
	if (A[index] % 2 == 0) {
		result[index] = A[index] + B[index];
	} 
	else {
		result[index] = A[index] - B[index];
	}
}

void do_work(bool useBranching) {
	const int array_size_bytes = sizeof(int) * N;
	srand(time(NULL));


	// populate arrays A, B
	int *a = (int*)malloc(array_size_bytes);
	int *b = (int*)malloc(array_size_bytes);
	int *res = (int*)malloc(array_size_bytes);
	for (int i=0; i<N; i++) {
		a[i] = rand();
		b[i] = rand();
	}

	// execute CPU code
	clock_t start, end;
	if (!useBranching) {
		start = clock();
		for (int i=0; i<N; i++) {
			add(a, b, res, i);
		}
		end = clock();
	}
	else {
		start = clock();
		for (int i=0; i<N; i++) {
			add_w_branch(a, b, res, i);
		}
		end = clock();
	}

	double time_ms = 1000.0 * (end - start)/CLOCKS_PER_SEC;

	printf("\nCPU execution time (%s): %.3f ms", (useBranching ? "branching" : "non-branching"), time_ms);

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

	printf("run cpu version (%s) with totalThreads %d, blockSize %d, numBlocks %d\n", (useBranching ? "branching" : "non-branching"), totalThreads, blockSize, numBlocks);
	do_work(useBranching);
}
