/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "assert.h"

static const int WORK_SIZE = 256;

typedef unsigned short int u16;
typedef unsigned int u32;

#define KERNEL_LOOP 4096

__constant__ u32 const_data_gpu[KERNEL_LOOP];
__device__ static u32 gmem_data_gpu[KERNEL_LOOP];
static u32 const_data_host[KERNEL_LOOP];

__global__ void const_test_gpu_gmem(u32 * const data, const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		u32 d = gmem_data_gpu[0];

		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d ^= gmem_data_gpu[0];
			d |= gmem_data_gpu[1];
			d &= gmem_data_gpu[2];
			d |= gmem_data_gpu[3];
		}

		data[tid] = d;
	}
}


__global__ void const_test_gpu_const(u32 * const data, const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		u32 d = const_data_gpu[0];

		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d ^= const_data_gpu[0];
			d |= const_data_gpu[1];
			d &= const_data_gpu[2];
			d |= const_data_gpu[3];
		}

		data[tid] = d;
	}
}

__host__ void wait_exit(void)
{
	char ch;

	printf("\nPress any key to exit");
	ch = getchar();
}

__host__ void cuda_error_check(const char * prefix, const char * postfix)
{
	if(cudaPeekAtLastError() != cudaSuccess)
	{
		printf("\n%s%s%s",prefix,cudaGetErrorString(cudaGetLastError()),postfix);
		cudaDeviceReset();
		wait_exit();
		exit(1);
	}
}

__host__ void generate_rand_data(u32 * host_data_ptr)
{
	for(u32 i=0; i < KERNEL_LOOP; i++)
	{
		host_data_ptr[i] = (u32) rand();
	}
}

__host__ void gpu_kernel(void)
{
	const u32 num_elements = (128*1024);
	const u32 num_threads = 256;
	const u32 num_blocks = (num_elements + (num_threads-1))/num_threads;
	const u32 num_bytes = num_elements * sizeof(u32);
	int max_device_num;
	const int max_runs = 6;

	cudaGetDeviceCount(&max_device_num);

	for(int device_num=0; device_num < max_device_num; device_num++)
	{
		cudaSetDevice(device_num);

		u32 * data_gpu;
		cudaEvent_t kernel_start1, kernel_stop1;
		cudaEvent_t kernel_start2, kernel_stop2;
		float delta_time1 = 0.0F, delta_time2 = 0.0F;
		struct cudaDeviceProp device_prop;
		char device_prefix[261];

		cudaMalloc(&data_gpu, num_bytes);
		cudaEventCreate(&kernel_start1);
		cudaEventCreate(&kernel_start2);
		cudaEventCreateWithFlags(&kernel_stop1, cudaEventBlockingSync);
		cudaEventCreateWithFlags(&kernel_stop2, cudaEventBlockingSync);

		cudaGetDeviceProperties(&device_prop, device_num);
		sprintf(device_prefix, "ID: %d %s:", device_num, device_prop.name);

		for(int num_test=0; num_test < max_runs; num_test++)
		{
			generate_rand_data(const_data_host);

			cudaMemcpyToSymbol(const_data_gpu, const_data_host, KERNEL_LOOP * sizeof(u32));

			const_test_gpu_gmem <<<num_blocks, num_threads>>>(data_gpu, num_elements);
			cuda_error_check("Error ", " returned from literal runtime  kernel!");

			cudaEventRecord(kernel_start1,0);

			const_test_gpu_gmem <<<num_blocks, num_threads>>>(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal runtime  kernel!");

			cudaEventRecord(kernel_stop1,0);
			cudaEventSynchronize(kernel_stop1);
			cudaEventElapsedTime(&delta_time1, kernel_start1, kernel_stop1);

			cudaMemcpyToSymbol(gmem_data_gpu, const_data_host, KERNEL_LOOP * sizeof(u32));
			const_test_gpu_const<<< num_blocks, num_threads >>>(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal startup  kernel!");

			cudaEventRecord(kernel_start2,0);

			const_test_gpu_const<<< num_blocks, num_threads >>>(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal startup  kernel!");

			cudaEventRecord(kernel_stop2,0);
			cudaEventSynchronize(kernel_stop2);
			cudaEventElapsedTime(&delta_time2, kernel_start2, kernel_stop2);

			if(delta_time1 > delta_time2)
			{
				printf("\n%sConstant version is faster by: %.2fms (G=%.2fms vs. C=%.2fms)",device_prefix, delta_time1-delta_time2, delta_time1, delta_time2);
			}
			else
			{
				printf("\n%sGMEM version is faster by: %.2fms (G=%.2fms vs. C=%.2fms)",device_prefix, delta_time2-delta_time1, delta_time1, delta_time2);
			}

		}

		cudaEventDestroy(kernel_start1);
		cudaEventDestroy(kernel_start2);
		cudaEventDestroy(kernel_stop1);
		cudaEventDestroy(kernel_stop2);
		cudaFree(data_gpu);

		cudaDeviceReset();
		printf("\n");
	}
	wait_exit();
}

__host__ __device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{
	u32 *d = NULL;
	int i;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (i = 0; i < WORK_SIZE; i++){
		idata[i] = (unsigned int) i;
	}

	cudaMalloc((void**) &d, sizeof(int) * WORK_SIZE);
	cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice);

	const_test_gpu_const<<<1, WORK_SIZE>>>(d,WORK_SIZE);

	cudaThreadSynchronize();	// Wait for the GPU launched work to complete
	cudaGetLastError();
	cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost);

	for (i = 0; i < WORK_SIZE; i++){
		printf("Input value: %u, device output: %u, host output: %u\n",idata[i], odata[i], bitreverse(idata[i]));
	}	

	cudaFree((void*) d);
	cudaDeviceReset();


}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	execute_host_functions();
	execute_gpu_functions();

	return 0;
}
