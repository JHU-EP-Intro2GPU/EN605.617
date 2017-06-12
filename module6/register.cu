#include <stdio.h>
#include <stdlib.h>

#define KERNEL_LOOP 128

typedef unsigned short int u16;
typedef unsigned int u32;
typedef unsigned long long int u128;

__device__ static unsigned int d_tmp = 0;

__device__ u128 packed_array[KERNEL_LOOP];

static u128 host_packed_array[KERNEL_LOOP];
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

#define CUDA_CALL(x) {														\
	cudaError_t _m_cudaStat = x;											\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


__host__ void wait_exit(void)
{
	char ch;

	printf("\nPress any key to exit");
	ch = getchar();
}

__host__ void generate_rand_data(u128 * host_data_ptr)
{
	for(u128 i=0; i < KERNEL_LOOP; i++)
	{
		host_data_ptr[i] = (u128) rand();
	}
}

__global__ void test_gpu_register(u32 * const data, const u32 num_elements)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		unsigned int d_tmp = 0;

		for(int i=0; i < KERNEL_LOOP; i++)
		{
			d_tmp |= (packed_array[i] << i);
		}
		data[tid] = d_tmp;
	}
}

__global__ void test_gpu_gmem(u32 * const data, const u32 num_elements)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		for(int i=0; KERNEL_LOOP;i++)
		{
			d_tmp |= (packed_array[i] << i);
		}
		
		data[tid] = d_tmp;
	}
}

__host__ void gpu_kernel(void)
{
	const u32 num_elements = (KERNEL_LOOP*1024);
	const u32 num_threads = 256;
	const u32 num_blocks = (num_elements + (num_threads-1))/num_threads;
	const u32 num_bytes = num_elements * sizeof(u32);

	u32 * data_gpu;

	CUDA_CALL(cudaMalloc(&data_gpu, num_bytes));

	generate_rand_data(host_packed_array);

	CUDA_CALL(cudaMemcpyToSymbol(packed_array, host_packed_array, KERNEL_LOOP * sizeof(u128)));

	test_gpu_register <<<num_blocks, num_threads>>>(data_gpu, num_elements);
	wait_exit();
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{

}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	gpu_kernel();
}
