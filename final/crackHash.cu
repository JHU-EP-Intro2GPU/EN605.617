// EN.605.417
// Bill Burgett

#include <stdio.h>
#include "md5.h"

#define WORK_SIZE 128
#define	HASH_SIZE 128

#define TESTMAX 

__global__ void deviceHash(MD5 md5, String *crackedPass, int maxSize, String charSet, int charSize)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	char* tryPass[maxSize];
	if (tid < TESTMAX){
		for(int i=(maxSize-1);i>=0;i--){
			tryPass[i]=charSet[(tid%((int)pow( (double)charSize, (double)i )))];
		}
		md5.digestString(tryPass));
	}
}

float runHash();
{
	const unsigned int num_threads = 128;
	const unsigned int num_blocks = WORK_SIZE / num_threads;
	float delta;
 	String *crackedPass;
	const MD5 md5;
	const unsigned int max = 4000;
	const String charSet = "abcdefghijklmnopqrstuvwxyzABCEDFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()";
 
	cudaEvent_t start_time = get_time();
 	
 	cudaMalloc(&crackedPass, HASH_SIZE);
	
 	cudaMemcpyToSymbol(dArray_a, &d_a, (HASH_SIZE*WORK_SIZE));
	cudaMemcpyToSymbol(dArray_b, &d_b, (HASH_SIZE*WORK_SIZE));

	deviceHash << <num_blocks, num_threads >> >(max, md5, crackedPass);
	
	cudaThreadSynchronize();	// Wait for the GPU launched work to complete
	cudaGetLastError();

	//stop the clock
	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	//find execution time
	cudaEventElapsedTime(&delta, start_time, end_time);

	cudaFree(crackedPass);
	cudaDeviceReset();

	return delta;
}

int main()
{
	float elapsedTime = runHash();
	printf("Execution time = %f\n",elapsedTime);	

	return 0;
}
