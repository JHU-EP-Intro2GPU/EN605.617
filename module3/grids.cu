#include <stdio.h>

__global__ void what_is_my_id_2d_A(
				unsigned int * const block_x,
				unsigned int * const block_y,
				unsigned int * const thread,
				unsigned int * const calc_thread,
				unsigned int * const x_thread,
				unsigned int * const y_thread,
				unsigned int * const grid_dimx,
				unsigned int * const block_dimx,
				unsigned int * const grid_dimy,
				unsigned int * const block_dimy)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

	block_x[thread_idx] = blockIdx.x;
	block_y[thread_idx] = blockIdx.y;
	thread[thread_idx] = threadIdx.x;
	calc_thread[thread_idx] = thread_idx;
	x_thread[thread_idx] = idx;
	y_thread[thread_idx] = idy;
	grid_dimx[thread_idx] = gridDim.x;
	block_dimx[thread_idx] = blockDim.x;
	grid_dimy[thread_idx] = gridDim.y;
	block_dimy[thread_idx] = blockDim.y;
}

// #define ARRAY_SIZE_X 32
// #define ARRAY_SIZE_& 16

// #define ARRAY_SIZE_IN_BYTES ((array_size_x) * (array_size_y) * (sizeof(unsigned int)))

/* Declare statically six arrays of ARRAY_SIZE each */

// assumption is that threads_rect.x * blocks_rect.x = threads_square.x * blocks_square.x
// assumption is that threads_rect.y * blocks_rect.y = threads_square.y * blocks_square.y
void main_sub( const dim3 threads_rect, const dim3 blocks_rect, const dim3 threads_square,  const dim3 blocks_square)
{
	/* Total thread count = 32 * 4 = 128 */
	// const dim3 threads_rect(32,4);
	// const dim3 blocks_rect(1,4);

	/* Total thread count = 16 * 8 = 128 */
	// const dim3 threads_square(16, 8); /* 16 * 8 */
	// const dim3 blocks_square(2,2);
	unsigned int array_size_x = threads_rect.x * blocks_rect.x;
	unsigned int array_size_y = threads_rect.y * blocks_rect.y;
	unsigned int array_size_bytes = array_size_x * array_size_y  * sizeof(unsigned int);
	/* Needed to wait for a character at exit */
	char ch;
	unsigned int cpu_block_x[array_size_y][array_size_x];
	unsigned int cpu_block_y[array_size_y][array_size_x];
	unsigned int cpu_thread[array_size_y][array_size_x];
	unsigned int cpu_warp[array_size_y][array_size_x];
	unsigned int cpu_calc_thread[array_size_y][array_size_x];
	unsigned int cpu_xthread[array_size_y][array_size_x];
	unsigned int cpu_ythread[array_size_y][array_size_x];
	unsigned int cpu_grid_dimx[array_size_y][array_size_x];
	unsigned int cpu_block_dimx[array_size_y][array_size_x];
	unsigned int cpu_grid_dimy[array_size_y][array_size_x];
	unsigned int cpu_block_dimy[array_size_y][array_size_x];
	/* Declare statically six arrays of ARRAY_SIZE each */
	unsigned int * gpu_block_x;
	unsigned int * gpu_block_y;
	unsigned int * gpu_thread;
	unsigned int * gpu_warp;
	unsigned int * gpu_calc_thread;
	unsigned int * gpu_xthread;
	unsigned int * gpu_ythread;
	unsigned int * gpu_grid_dimx;
	unsigned int * gpu_block_dimx;
	unsigned int * gpu_grid_dimy;
	unsigned int * gpu_block_dimy;

	/* Allocate arrays on the GPU */
	cudaMalloc((void **)&gpu_block_x, array_size_bytes);
	cudaMalloc((void **)&gpu_block_y, array_size_bytes);
	cudaMalloc((void **)&gpu_thread, array_size_bytes);
	cudaMalloc((void **)&gpu_warp, array_size_bytes);
	cudaMalloc((void **)&gpu_calc_thread, array_size_bytes);
	cudaMalloc((void **)&gpu_xthread, array_size_bytes);
	cudaMalloc((void **)&gpu_ythread, array_size_bytes);
	cudaMalloc((void **)&gpu_grid_dimx, array_size_bytes);
	cudaMalloc((void **)&gpu_block_dimx, array_size_bytes);
	cudaMalloc((void **)&gpu_grid_dimy, array_size_bytes);
	cudaMalloc((void **)&gpu_block_dimy, array_size_bytes);

	for( int kernel= 0; kernel < 2; kernel++)
	{
		switch(kernel)
		{
			case 0:
			{
				/* Execute our kernel */
				what_is_my_id_2d_A<<<blocks_rect, threads_rect>>>(gpu_block_x, gpu_block_y,
	gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
	gpu_grid_dimy, gpu_block_dimy);
			} break;

			case 1:
			{
				/* Execute our kernel */
				what_is_my_id_2d_A<<<blocks_square, threads_square>>>(gpu_block_x, gpu_block_y,
	gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
	gpu_grid_dimy, gpu_block_dimy);
			} break;

			default: exit(1); break;
		}

		/* Copy back the gpu results to the CPU */
		cudaMemcpy(cpu_block_x, gpu_block_x, array_size_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_y, gpu_block_y, array_size_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_thread, gpu_thread, array_size_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_calc_thread, gpu_calc_thread, array_size_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_xthread, gpu_xthread, array_size_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_ythread, gpu_ythread, array_size_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, array_size_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_dimx, gpu_block_dimx, array_size_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, array_size_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_dimy, gpu_block_dimy, array_size_bytes, cudaMemcpyDeviceToHost);

		printf("\nKernel %d\n", kernel);
		/* Iterate through the arrays and print */
		for(int y = 0; y < array_size_y; y++)
		{
			for(int x = 0; x < array_size_x; x++)
			{
				printf("CT: %2u BKX: %1u BKY: %1u TID: %2u YTID: %2u XTID: %2u GDX: %1u BDX: %1u GDY: %1u BDY: %1u\n",
						cpu_calc_thread[y][x], cpu_block_x[y][x], cpu_block_y[y][x], cpu_thread[y][x], cpu_ythread[y][x],
						cpu_xthread[y][x], cpu_grid_dimx[y][x], cpu_block_dimx[y][x], cpu_grid_dimy[y][x], cpu_block_dimy[y][x]);

			}
		}
	}

	/* Free the arrays on the GPU as now we're done with them */
	cudaFree(gpu_block_x);
	cudaFree(gpu_block_y);
	cudaFree(gpu_thread);
	cudaFree(gpu_warp);
	cudaFree(gpu_calc_thread);
	cudaFree(gpu_xthread);
	cudaFree(gpu_ythread);
	cudaFree(gpu_grid_dimx);
	cudaFree(gpu_block_dimx);
	cudaFree(gpu_grid_dimy);
	cudaFree(gpu_block_dimy);
}



int main()
{
	// 128 total threads
	main_sub(
		dim3(8, 4),    // 32 threads/block
		dim3(1, 4),    // 4 blocks
		dim3(4, 8),    // 32 threads/block
		dim3(2, 2));   // 4 blocks

	// 256 total threads
	main_sub(
		dim3(16, 4),   // 64 threads/block
		dim3(1, 4),    // 4 blocks
		dim3(8, 8),    // 64 threads/block
		dim3(2, 2));   // 4 blocks

	// 384 total threads
	main_sub(
		dim3(24, 4),   // 96 threads/block
		dim3(1, 4),    // 4 blocks
		dim3(12, 8),   // 96 threads/block
		dim3(2, 2));   // 4 blocks

	// 512 total threads
	main_sub(
		dim3(32, 4),   // 128 threads/block
		dim3(1, 4),    // 4 blocks
		dim3(16, 8),   // 128 threads/block
		dim3(2, 2));   // 4 blocks

	// 1024 total threads
	main_sub(
		dim3(32, 8),   // 256 threads/block
		dim3(1, 4),    // 4 blocks
		dim3(32, 8),   // 256 threads/block
		dim3(2, 2));   // 4 blocks

	return EXIT_SUCCESS;
}