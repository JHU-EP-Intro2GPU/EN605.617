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
#include <assert.h>

#define KERNEL_LOOP 65536

#define WORK_SIZE 256;

typedef unsigned short int u16;
typedef unsigned int u32;

__constant__  static const unsigned int const_data_01 = 0x55555555;
__constant__  static const unsigned int const_data_02 = 0x77777777;
__constant__  static const unsigned int const_data_03 = 0x33333333;
__constant__  static const unsigned int const_data_04 = 0x11111111;

#define NUM_ELEMENTS 256

#define CUDA_CALL(x) 														\
{																			\
	cudaError_t _m_cudaStat = x;											\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__global__ void const_test_gpu_literal(u32 * data,
		const u32 num_elements) {
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < num_elements) {
		u32 d = 0x55555555;

		for (int i = 0; i < KERNEL_LOOP; i++) {
			d ^= 0x55555555;
			d |= 0x77777777;
			d &= 0x33333333;
			d |= 0x11111111;
		}

		data[tid] = d;
	}
}

__global__ void const_test_gpu_const(unsigned int * const data, const unsigned int num_elements) {
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < num_elements) {
		unsigned int d = const_data_01;

		for (int i = 0; i < KERNEL_LOOP; i++) {
			d ^= const_data_01;
			d |= const_data_02;
			d &= const_data_03;
			d |= const_data_04;
		}

		data[tid] = d;
	}
}

__host__ void gpu_kernel(void) {
	const unsigned int num_elements = (128 * 1024);
	const unsigned int num_threads = 256;
	const unsigned int num_blocks = (num_elements + (num_threads - 1)) / num_threads;
	const unsigned int num_bytes = num_elements * sizeof(unsigned int);
	int max_device_num;
	const int max_runs = 6;

	CUDA_CALL(cudaGetDeviceCount(&max_device_num));

	for (int device_num = 0; device_num < max_device_num; device_num++) {
		CUDA_CALL(cudaSetDevice(device_num));

		for (int num_test = 0; num_test < max_runs; num_test++) {
			unsigned int * data_gpu;
			cudaEvent_t kernel_start1, kernel_stop1;
			cudaEvent_t kernel_start2, kernel_stop2;
			float delta_time1 = 0.0f, delta_time2 = 0.0F;
			struct cudaDeviceProp device_prop;
			char device_prefix[261];

			CUDA_CALL(cudaMalloc(&data_gpu, num_bytes));
			CUDA_CALL(cudaEventCreate(&kernel_start1));
			CUDA_CALL(cudaEventCreate(&kernel_start2));
			CUDA_CALL(
					cudaEventCreateWithFlags(&kernel_stop1,
							cudaEventBlockingSync));
			CUDA_CALL(
					cudaEventCreateWithFlags(&kernel_stop2,
							cudaEventBlockingSync));

			CUDA_CALL(cudaGetDeviceProperties(&device_prop, device_num));
			sprintf(device_prefix, "ID: %d %s:", device_num, device_prop.name);

			const_test_gpu_literal<<<num_blocks, num_threads>>>(data_gpu,
					num_elements);

//			cuda_error_check("Error ",
//					" returned from literal startup  kernel!");

			CUDA_CALL(cudaEventRecord(kernel_start1, 0));
			const_test_gpu_literal<<<num_blocks, num_threads>>>(data_gpu,
					num_elements);

//			cuda_error_check("Error ",
//					" returned from literal runtime  kernel!");

			CUDA_CALL(cudaEventRecord(kernel_stop1, 0));
			CUDA_CALL(cudaEventSynchronize(kernel_stop1));
			CUDA_CALL(
					cudaEventElapsedTime(&delta_time1, kernel_start1,
							kernel_stop1));

			const_test_gpu_const<<<num_blocks, num_threads>>>(data_gpu,
					num_elements);

//			cuda_error_check("Error ",
//					" returned from literal startup  kernel!");

			CUDA_CALL(cudaEventRecord(kernel_stop2, 0));
			CUDA_CALL(cudaEventSynchronize(kernel_stop2));
			CUDA_CALL(
					cudaEventElapsedTime(&delta_time2, kernel_start2,
							kernel_stop2));

			if (delta_time1 > delta_time2) {
				printf(
						"\n%sConstant version is faster by: %.2fms (Const=%.2fms vs. Literal=%.2fms)",
						device_prefix, delta_time1 - delta_time2, delta_time1,
						delta_time2);
			} else {
				printf(
						"\n%sLiteral version is faster by: %.2fms (Const=%.2fms vs. Literal=%.2fms)",
						device_prefix, delta_time2 - delta_time1, delta_time1,
						delta_time2);
			}

			CUDA_CALL(cudaEventDestroy(kernel_start1));
			CUDA_CALL(cudaEventDestroy(kernel_start2));
			CUDA_CALL(cudaEventDestroy(kernel_stop1));
			CUDA_CALL(cudaEventDestroy(kernel_stop2));
			CUDA_CALL(cudaFree(data_gpu));
		}

		CUDA_CALL(cudaDeviceReset());
		printf("\n");
	}
//	wait_exit();
}

__device__  static unsigned int data_01 = 0x55555555;
__device__  static unsigned int data_02 = 0x77777777;
__device__  static unsigned int data_03 = 0x33333333;
__device__  static unsigned int data_04 = 0x11111111;

__global__ void const_test_gpu_gmem(unsigned int * const data, const unsigned int num_elements) {
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < num_elements) {
		unsigned int d = data_01;

		for (int i = 0; i < KERNEL_LOOP; i++) {
			d ^= data_01;
			d |= data_02;
			d &= data_03;
			d |= data_04;
		}

		data[tid] = d;
	}
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{
	const u32 num_threads = 256;
	const u32 num_blocks = NUM_ELEMENTS/num_threads;

	u32 data[NUM_ELEMENTS];
	const_test_gpu_literal<<<num_blocks,num_threads>>>(data, NUM_ELEMENTS);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	execute_host_functions();
	execute_gpu_functions();

	return 0;
}
