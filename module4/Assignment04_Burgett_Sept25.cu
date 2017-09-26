/* *
 * EN.605.417.FA17 - Bill Burgett
 *
 * This program is to test the functionality and compare execution times
 * of host and global memory. Specifically, it utilizes both pageable
 * and pinned memory on the host, and copies the specified memory to the
 * device for execution. Enable debug to compare outputs of the device 
 * against outputs from the host to verify that the functions executed 
 * correctly. The inputs are fairly simple with the first being an array
 * counting from 0 to WORK_SIZE, and the second input is just double the first.
 * The function operating on both the host and the device is simply adding
 * these arrays together.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

//set the size of the input arrays
static const int WORK_SIZE = 4096;

/*
* function can be called from host or kernel to perform addition.
*/
__host__ __device__ unsigned int devAdd(unsigned int num_a, unsigned int num_b) {
	return num_a+num_b;
}

/*
* Helper function to get the time for execution length comparisons.
* source: Module 4 Activity Files: global_memory.cu
* https://github.com/JHU-EP-Intro2GPU/EN605.417.FA/blob/master/module4/global_memory.cu
*/
__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

/*
 * CUDA kernel function that adds the elements from two arrays and stores result in the first.
 */
__global__ void devAdd(void *d_a, void *d_b) {
	unsigned int *id_a = (unsigned int*) d_a;
	unsigned int *id_b = (unsigned int*) d_b;
	id_a[threadIdx.x] = devAdd(id_a[threadIdx.x],id_b[threadIdx.x]);
}

/*
 * Host function to create the paged memory and execute the kernel function
 */
float PagedAdd(int debugFlag)
{
	void *d_a = NULL;
	void *d_b = NULL;
	unsigned int idata_a[WORK_SIZE], idata_b[WORK_SIZE], odata[WORK_SIZE];
	int i;

	//device
	cudaMalloc((void**)&d_a, sizeof(int) * WORK_SIZE);
	cudaMalloc((void**)&d_b, sizeof(int) * WORK_SIZE);

	//fill input data arrays
	for (i = 0; i < WORK_SIZE; i++){
		idata_a[i] = (unsigned int)i;
		idata_b[i] = (unsigned int)i * 2;
	}	

	//start the clock
	cudaEvent_t start_time = get_time();

	//transfer paged memory to device
	cudaMemcpy(d_a, idata_a, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, idata_b, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice);
	
	//do the addition
	devAdd<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d_a,d_b);

	//wait for the GPU launched work to complete
	cudaThreadSynchronize();	
	cudaGetLastError();

	//stop the clock
	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	//find execution time
	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);
	
	//get result
	cudaMemcpy(odata, d_a, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost);

	//is debug on?
	if (debugFlag == 1) {
		//Print input, device result, and do the result on the host as well to check
		for (i = 0; i < WORK_SIZE; i++)
			printf("Pageable - Input values: %u + %u, device output: %u, host output: %u\n", idata_a[i], idata_b[i], odata[i], devAdd(idata_a[i], idata_b[i]));
	}

	cudaFree((void* ) d_a);
	cudaFree((void* ) d_b);
	cudaDeviceReset();

	return delta;
}

/*
* Host function to create the paged memory and execute the kernel function
*
* This was modified from the sample on slide 11 of the mod 4 video slides (Module4AHostMemory.ppt)
* source: Mod 4 video - CUDA Host Memory
* https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc
*/
float PinnedAdd(int debugFlag) {
	//unsigned int nElements = 4 * 1024 * 1024;
	const unsigned int bytes = WORK_SIZE * sizeof(int);

	unsigned int *h_aPageable, *h_bPageable;
	unsigned int *h_aPinned, *h_bPinned;
	unsigned int *d_a, *d_b;
	unsigned int *h_Result;
	int i;

	//result
	h_Result = (unsigned int*)malloc(bytes);
	
	//host pageable
	h_aPageable = (unsigned int*)malloc(bytes);
	h_bPageable = (unsigned int*)malloc(bytes);

	//host pinned
	cudaMallocHost((void**)&h_aPinned, bytes);
	cudaMallocHost((void**)&h_bPinned, bytes);

	//device
	cudaMalloc((void**)&d_a, bytes);
	cudaMalloc((void**)&d_b, bytes);

	//fill input data arrays
	for (int i = 0; i < WORK_SIZE; i++) {
		h_aPageable[i] = (unsigned int)i;
		h_bPageable[i] = (unsigned int)i * 2;
	}

	//fill pinned memory
	memcpy(h_aPinned, h_aPageable, bytes);
	memcpy(h_bPinned, h_bPageable, bytes);

	//start the clock
	cudaEvent_t start_time = get_time();

	//transfer pinned memory to device
	cudaMemcpy(d_a, h_aPinned, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_bPinned, bytes, cudaMemcpyHostToDevice);

	//do the addition
	devAdd << <1, WORK_SIZE, WORK_SIZE * sizeof(int) >> >(d_a, d_b);

	//wait for the GPU launched work to complete
	cudaThreadSynchronize();
	cudaGetLastError();

	//stop the clock
	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	//find execution time
	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);

	//get device result
	cudaMemcpy(h_Result, d_a, bytes, cudaMemcpyDeviceToHost);

	//is debug on?
	if (debugFlag == 1) {
		//Print input, device result, and do the result on the host as well to check
		for (i = 0; i < WORK_SIZE; i++)
			printf("Pinned - Input values: %u + %u, device output: %u, host output: %u\n", h_aPinned[i], h_bPinned[i], h_Result[i], devAdd(h_aPinned[i], h_bPinned[i]));
	}

	//free memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFreeHost(h_aPinned);
	cudaFreeHost(h_bPinned);
	free(h_aPageable);
	free(h_bPageable);
	free(h_Result);

	return delta;
}

/*
* main class to execute the above functions
*/
int main(void) {
	
	int userInput, debugFlag=0;
	cout << "Enter 1 to run with debug on, else enter any number: ";
	cin >> userInput;
	cout << "";

	if (userInput == 1)debugFlag = 1;

	float durationPinned = PinnedAdd(debugFlag);
	float durationPaged = PagedAdd(debugFlag);

	printf("Paged time  = %fms\nPinned time = %fms\n", durationPaged, durationPinned);

	return 0;
}