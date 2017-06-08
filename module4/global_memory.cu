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

static const int WORK_SIZE = 256;

#define NUM_ELEMENTS 4096

typedef struct {
	unsigned int a;
	unsigned int b;
	unsigned int c;
	unsigned int d;
} INTERLEAVED_T;

typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[NUM_ELEMENTS];

typedef unsigned int ARRAY_MEMBER_T[NUM_ELEMENTS];

typedef struct {
	ARRAY_MEMBER_T a;
	ARRAY_MEMBER_T b;
	ARRAY_MEMBER_T c;
	ARRAY_MEMBER_T d;
} NON_INTERLEAVED_T;

__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

__host__ float add_test_non_interleaved_cpu(
		NON_INTERLEAVED_T host_dest_ptr,
		NON_INTERLEAVED_T const host_src_ptr, const unsigned int iter,
		const unsigned int num_elements) {

	cudaEvent_t start_time = get_time();

	for (unsigned int tid = 0; tid < num_elements; tid++) {
		for (unsigned int i = 0; i < iter; i++) {
			host_dest_ptr.a[tid] += host_src_ptr.a[tid];
			host_dest_ptr.b[tid] += host_src_ptr.b[tid];
			host_dest_ptr.c[tid] += host_src_ptr.c[tid];
			host_dest_ptr.d[tid] += host_src_ptr.d[tid];
		}
	}

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);

	return delta;
}

__host__ float add_test_interleaved_cpu(INTERLEAVED_T * const host_dest_ptr,
		const INTERLEAVED_T * const host_src_ptr, const unsigned int iter,
		const unsigned int num_elements) {
	cudaEvent_t start_time = get_time();
	for (unsigned int tid = 0; tid < num_elements; tid++) {
		printf("tid: %u ", tid);
		for (unsigned int i = 0; i < iter; i++) {
			printf("iteration: %un", iter);
			host_dest_ptr[tid].a += host_src_ptr[tid].a;
			host_dest_ptr[tid].b += host_src_ptr[tid].b;
			host_dest_ptr[tid].c += host_src_ptr[tid].c;
			host_dest_ptr[tid].d += host_src_ptr[tid].d;
		}
	}

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);

	return delta;
}

__global__ void add_kernel_interleaved(INTERLEAVED_T * const dest_ptr,
		const INTERLEAVED_T * const src_ptr, const unsigned int iter,
		const unsigned int num_elements) {

	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(tid < num_elements)
	{
		for(unsigned int i=0; i<iter; i++)
		{
			dest_ptr[tid].a += src_ptr[tid].a;
			dest_ptr[tid].b += src_ptr[tid].b;
			dest_ptr[tid].c += src_ptr[tid].c;
			dest_ptr[tid].d += src_ptr[tid].d;
		}
	}
}

__global__ void add_kernel_non_interleaved(
		NON_INTERLEAVED_T * const dest_ptr,
		NON_INTERLEAVED_T * const src_ptr, const unsigned int iter,
		const unsigned int num_elements) {

	for (unsigned int tid = 0; tid < num_elements; tid++) {
		for (unsigned int i = 0; i < iter; i++) {
			dest_ptr->a[tid] += src_ptr->a[tid];
			dest_ptr->b[tid] += src_ptr->b[tid];
			dest_ptr->c[tid] += src_ptr->c[tid];
			dest_ptr->d[tid] += src_ptr->d[tid];
		}
	}
}

__host__ float add_test_interleaved(INTERLEAVED_T * const host_dest_ptr,
		const INTERLEAVED_T * const host_src_ptr, const unsigned int iter,
		const unsigned int num_elements)
{
	const unsigned int num_threads = 256;
	const unsigned int num_blocks = (num_elements + (num_threads-1)) / num_threads;

	const size_t num_bytes = (sizeof(INTERLEAVED_T) * num_elements);
	INTERLEAVED_T * device_dest_ptr;
	INTERLEAVED_T * device_src_ptr;

	cudaMalloc((void **) &device_src_ptr, num_bytes);
	cudaMalloc((void **) &device_dest_ptr, num_bytes);

	cudaEvent_t kernel_start, kernel_stop;
	cudaEventCreate(&kernel_start,0);
	cudaEventCreate(&kernel_stop,0);

	cudaStream_t test_stream;
	cudaStreamCreate(&test_stream);

	cudaMemcpy(device_src_ptr, host_src_ptr, num_bytes,cudaMemcpyHostToDevice);

	cudaEventRecord(kernel_start, 0);

	add_kernel_interleaved<<<num_blocks,num_threads>>>(device_dest_ptr, device_src_ptr, iter, num_elements);

	cudaEventRecord(kernel_stop, 0);

	cudaEventSynchronize(kernel_stop);

	float delta = 0.0F;
	cudaEventElapsedTime(&delta, kernel_start, kernel_stop);

	cudaFree(device_src_ptr);
	cudaFree(device_dest_ptr);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
	cudaStreamDestroy(test_stream);

	return delta;
}

__host__ float select_samples_cpu(unsigned int * const sample_data,
									const unsigned int sample_interval,
									const unsigned int num_elements,
									const unsigned int * const src_data)
{
	cudaEvent_t kernel_start, kernel_stop;
	cudaEventCreate(&kernel_start,0);
	cudaEventCreate(&kernel_stop,0);

	cudaEventRecord(kernel_start, 0);

	unsigned int sample_idx = 0;

	for(unsigned int src_idx=0; src_idx<num_elements;src_idx+=sample_interval)
	{
		sample_data[sample_idx] = src_data[src_idx];
		sample_idx++;
	}

	cudaEventRecord(kernel_stop, 0);

	cudaEventSynchronize(kernel_stop);

	float delta = 0.0F;
	cudaEventElapsedTime(&delta, kernel_start, kernel_stop);
	return delta;
}

__global__ void select_samples_gpu_kernel(unsigned int * const sample_data,
											const unsigned int sample_interval,
											const unsigned int * const src_data)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	sample_data[tid] = src_data[tid*sample_interval];
}

__host__ float select_samples_gpu(unsigned int * const sample_data,
									const unsigned int sample_interval,
									const unsigned int num_elements,
									const unsigned int num_samples,
									const unsigned int * const src_data,
									const unsigned int num_threads_per_block,
									const char * prefix)
{
	const unsigned int num_blocks = num_samples / num_threads_per_block;

	assert((num_blocks * num_threads_per_block) == num_samples);

	cudaEvent_t kernel_start, kernel_stop;
	cudaEventCreate(&kernel_start,0);
	cudaEventCreate(&kernel_stop,0);

	cudaEventRecord(kernel_start, 0);

	select_samples_gpu_kernel<<<num_blocks, num_threads_per_block>>>(sample_data, sample_interval, src_data);

	cudaEventRecord(kernel_stop, 0);

	cudaEventSynchronize(kernel_stop);

	float delta = 0.0F;
	cudaEventElapsedTime(&delta, kernel_start, kernel_stop);
	return delta;
}

// simple comparison function found at http://www.tutorialspoint.com/c_standard_library/c_function_qsort.htm
int compare_func (const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}

__host__ float sort_samples_cpu(unsigned int * const sample_data,
									const unsigned int num_samples)
{
	cudaEvent_t kernel_start, kernel_stop;
	cudaEventCreate(&kernel_start,0);
	cudaEventCreate(&kernel_stop,0);

	cudaEventRecord(kernel_start, 0);

	qsort(sample_data, num_samples, sizeof(unsigned int), &compare_func);

	cudaEventRecord(kernel_stop, 0);

	cudaEventSynchronize(kernel_stop);

	float delta = 0.0F;
	cudaEventElapsedTime(&delta, kernel_start, kernel_stop);
	return delta;
}


__host__ __device__ unsigned int bin_search3(const unsigned int * const src_data,
									const unsigned int search_value,
									const unsigned int num_elements)
{
	// Take the middle of the two sections
	unsigned int size = (num_elements >> 1);

	unsigned int start_idx = 0;
	bool found = false;

	do
	{
		const unsigned int src_idx = (start_idx+size);
		const unsigned int test_value = src_data[src_idx];

		if(test_value == search_value)
		{
			found = true;
		}
		else if(search_value > test_value)
		{
			start_idx = (start_idx+size);
		}

		if(found == false)
		{
			size >>= 1;
		}
	}
	while((found == false) && (size != 0));

	return (start_idx + size);
}

__host__ float count_bins_cpu(const unsigned int num_samples,
								const unsigned int num_elements,
								const unsigned int * const src_data,
								const unsigned int * const sample_data,
								unsigned int * const bin_count)
{
	cudaEvent_t start_time = get_time();

	for(unsigned int src_idx = 0; src_idx<num_elements;src_idx++)
	{
		const unsigned int data_to_find = src_data[src_idx];
		const unsigned int idx = bin_search3(sample_data,data_to_find,num_samples);
		bin_count[idx]++;
	}

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);
	return delta;
}

//Single data point atomic add to gmem
__global__ void count_bins_gpu_kernel5(const unsigned int num_samples,
										const unsigned int num_elements,
										const unsigned int * const src_data,
										const unsigned int * const sample_data,
										unsigned int * const bin_count)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	const unsigned int data_to_find = src_data[tid];

	const unsigned int idx = bin_search3(sample_data, data_to_find, num_samples);

	atomicAdd(&bin_count[idx],1);
}

__host__ float count_bins_gpu(const unsigned int num_samples,
										const unsigned int * const src_data,
										const unsigned int * const sample_data,
										unsigned int * const bin_count,
										const unsigned int num_threads,
										const char * prefix)
{

	const unsigned int num_blocks = num_samples / num_threads;

	cudaEvent_t start_time = get_time();

	count_bins_gpu_kernel5<<<num_blocks,num_threads>>>(num_samples, NUM_ELEMENTS, src_data, sample_data, bin_count);
//	cuda_error_check(prefix, "Error invoking count_bins_gpu_kernel");

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);
	return delta;
}

__host__ float calc_bin_idx_cpu(const unsigned int num_samples,
									const unsigned int * const bin_count,
									unsigned int * const dest_bin_idx)
{
	cudaEvent_t start_time = get_time();
	unsigned int prefix_sum = 0;
	for(unsigned int i = 0; i<num_samples;i++)
	{
		dest_bin_idx[i] = prefix_sum;
		prefix_sum += bin_count[i];
	}

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);
	return delta;
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
	INTERLEAVED_T host_dest_ptr[NUM_ELEMENTS];
	INTERLEAVED_T host_src_ptr[NUM_ELEMENTS];
	float duration = add_test_interleaved_cpu(host_dest_ptr, host_src_ptr, 4,NUM_ELEMENTS);
	printf("duration: %fmsn",duration);

}

void execute_gpu_functions()
{
	void *d = NULL;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];
	int i;
	for (i = 0; i < WORK_SIZE; i++)
		idata[i] = (unsigned int) i;

	cudaMalloc((void** ) &d, sizeof(int) * WORK_SIZE);
	
			cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE,
					cudaMemcpyHostToDevice);

	bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

	cudaThreadSynchronize();	// Wait for the GPU launched work to complete
	cudaGetLastError();
	
			cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE,
					cudaMemcpyDeviceToHost);

	for (i = 0; i < WORK_SIZE; i++)
		printf("Input value: %u, device output: %u, host output: %u\n",
				idata[i], odata[i], bitreverse(idata[i]));

	cudaFree((void* ) d);
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
