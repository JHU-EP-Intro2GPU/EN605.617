
#include <vector>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// Module 3
// Cuda kernels

// Add two 1-d arrays of Ts together
template<typename T>
__global__
void add_matrices(T dest_matrix[],
	T matrix_i[], T matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// dest_matrix[i] = threadIdx.x + matrix_j[i];
    dest_matrix[i] = matrix_i[i] + matrix_j[i];
}

// Subtract two 1-d arrays of Ts together
template<typename T>
__global__
void subtract_matrices(T dest_matrix[],
	T matrix_i[], T matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = matrix_i[i] - matrix_j[i];
}

// Multiply two 1-d arrays of Ts together
template<typename T>
__global__
void multiply_matrices(T dest_matrix[],
	T matrix_i[], T matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = matrix_i[i] * matrix_j[i];
}

// Modulo divide two 1-d arrays of Ts together
template<typename T>
__global__
void modulo_matrices(T dest_matrix[],
	T matrix_i[], T matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = matrix_i[i] % matrix_j[i];
}

// Modulo divide two 1-d arrays of Ts together
template<typename T>
__global__
void modulo_matrices_branched(T dest_matrix[],
	T matrix_i[], T matrix_j[])
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Replace any invalid modulo divisors with 0
    dest_matrix[i] = (matrix_j[i] > 0) ? matrix_i[i] % matrix_j[i] : 0;
}

////////////////////////////////////////////////////////////////////////////////
// Module 4

// Add a constant to an array
template<typename T, typename A>
__global__
void add_constant(T dest_matrix[],
	T matrix_i[], A offset)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest_matrix[i] = matrix_i[i] + offset;
}
////////////////////////////////////////////////////////////////////////////////
// Module 5

template<typename T>
inline void print_vector(const std::vector<T> & vec) {
	std::cout << "[";
    for (auto i : vec) {
        std::cout << +i << ",";
    }
	std::cout << "]\n";
}
