// EN.605.417
// Bill Burgett

#include <stdio.h>
#include "md5.h"

#define WORK_SIZE 128
#define	HASH_SIZE 128

/*
* This is the function that will actually perform the hashing on the GPU. It will use the thread id to
* determine the string to test. For proof-of-concept I have limited the character set to just integers for now.
* The thread id will correspond to a string built out of an input character set. For example, if the character
* set was 26 letters of the alphabet and the tid was 20000, then the resultant string would be "acoe"
*/
__device__ void deviceHash(MD5 md5, String *crackedPass, String charSet, int charSetSize)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int carry = tid;
	unsigned int selection;
	char* tryPass[maxSize];
	for(int i=charSetSize;i>1;i--){
		tryPass[i]=charSet[(carry/(Math.pow(charSetSize,i-1));
		carry=carry%(Math.pow(charSetSize,i-1)
	}
	tryPass[0]=charSet[carry];
	md5.digestString(tryPass));
}

//First cut at the wrapper to run the kernel
__global__ float runHash();
{
	const unsigned int num_threads = 128;
	const unsigned int num_blocks = WORK_SIZE / num_threads;
	float delta;
 	String *crackedPass;
	const MD5 md5;
	//const String charSet = "abcdefghijklmnopqrstuvwxyzABCEDFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()";
	const String charSet = "1234567890";
 
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
