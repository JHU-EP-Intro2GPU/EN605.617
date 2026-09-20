#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/*
// computes multiplicity of arrays and index and stores them in C
// input floating arrays A, B
// output array C 
*/
void arrayCalculation(
	float* A,
	float* B,
	float* C,
	int arraySize)
{
	for (int i = 0; i < arraySize; ++i) {
		C[i] = A[i] * B[i] * (i+1);
	}
}

/*
// computes sum of arrays A,B and stores them in C
// conditional branching is applied depending on whether A[i]> B[i]
// input floating arrays A, B
// output array C
*/
void arrayCalculationBranch(
	float* A,
	float* B,
	float* C,
	int arraySize)
{
	for (int i = 0; i < arraySize; ++i) {
		if (A[i] > B[i]) {
			// printf("Condition 1 Hit\n");
			C[i] = A[i] * B[i] * (i +1);
		}
		else {
			// printf("Condition 2 Hit\n");
			C[i] = (A[i] * B[i]) / (i +1);
		}
	}
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	printf("Total elements: %d\n", totalThreads);
	printf("Block size: %d\n", blockSize);
	printf("Number of blocks: %d\n\n", numBlocks);

	// allocate memory
	float* array_A = (float*)malloc(totalThreads * sizeof(float));
	float* array_B = (float*)malloc(totalThreads * sizeof(float));
	float* array_C = (float*)malloc(totalThreads * sizeof(float));
	float* CBranch = (float*)malloc(totalThreads * sizeof(float));

	// initialize input data
	printf("Initializing input...");
	for (int i = 0; i < totalThreads; ++i) {
		array_A[i] = (float)(i % 100) / 10.0f;
		array_B[i] = (float)((i + 50) % 100) / 10.0f;
	}

	// run vector calculation without branching
	clock_t start = clock(); //timer
	printf("running calculations without branching\n");
	arrayCalculation(array_A, array_B, array_C, totalThreads);

	clock_t end = clock(); // end timer no branching
	
	double cpuTime_NoBranch =
		((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

	// run calculations with branching
	start = clock(); // timer with branching
	printf("running with calculations\n");
	arrayCalculationBranch(array_A, array_B, CBranch, totalThreads);

	end = clock(); // end  timer with branching

	double cpuTime_Branching =
		((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
	
	// display result head
	printf("\nFirst 10 results:\n");

	for (int i = 0; i < 10 && i < totalThreads; ++i) {
		printf("array_C[%d] = %.2f    CBranch[%d] = %.2f\n",
			i, array_C[i], i, CBranch[i]);
	}
	// display performance results
	printf("CPU Performance\n");
	printf("-------------------------\n");
	printf("Without branching: %.4f ms\n", cpuTime_NoBranch);
	printf("With branching:    %.4f ms\n", cpuTime_Branching);



	// free memory
	free(array_A);
	free(array_B);
	free(array_C); // free array_C
	free(CBranch);

	return 0;
}