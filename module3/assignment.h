#include <vector>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// Cuda kernels

// Add two 1-d arrays of unsigned ints together
__global__
void add_matrices(unsigned int dest_matrix[],
	const unsigned int matrix_i[], const unsigned int matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// dest_matrix[i] = threadIdx.x + matrix_j[i];
    dest_matrix[i] = matrix_i[i] + matrix_j[i];
}

// Subtract two 1-d arrays of unsigned ints together
__global__
void subtract_matrices(unsigned int dest_matrix[],
	const unsigned int matrix_i[], const unsigned int matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = matrix_i[i] - matrix_j[i];
}

// Multiply two 1-d arrays of unsigned ints together
__global__
void multiply_matrices(unsigned int dest_matrix[],
	const unsigned int matrix_i[], const unsigned int matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = matrix_i[i] * matrix_j[i];
}

// Modulo divide two 1-d arrays of unsigned ints together
__global__
void modulo_matrices(unsigned int dest_matrix[],
	const unsigned int matrix_i[], const unsigned int matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = matrix_i[i] % matrix_j[i];
}

// Modulo divide two 1-d arrays of unsigned ints together
__global__
void modulo_matrices_branched(unsigned int dest_matrix[],
	const unsigned int matrix_i[], const unsigned int matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Replace any invalid modulo divisors with 0
    dest_matrix[i] = (matrix_j[i] > 0) ? matrix_i[i] % matrix_j[i] : 0;
}
////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline void print_vector(const std::vector<T> & vec) {
	std::cout << "[";
    for (auto i : vec) {
        std::cout << i << ",";
    }
	std::cout << "]\n";
}
