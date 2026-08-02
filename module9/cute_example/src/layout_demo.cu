/*
 * CuTe layout and tensor basics.
 *
 * Demonstrates Layout creation, coordinate-to-index mapping, and a simple
 * elementwise GPU kernel using CuTe tensors.
 */

#include <iostream>
#include <vector>
#include <cute/tensor.hpp>
#include "cute_helper.h"

using namespace cute;

void demo_layouts() {
  std::cout << "=== CuTe Layout Demo ===\n\n";

  auto row_major = make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutRight{});
  std::cout << "Row-major 4x8 layout:\n";
  print(row_major);
  std::cout << "\n\n";

  auto col_major = make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutLeft{});
  std::cout << "Column-major 4x8 layout:\n";
  print(col_major);
  std::cout << "\n\n";

  std::cout << "Row-major index(2,3) = " << row_major(2, 3) << "\n";
  std::cout << "Col-major index(2,3) = " << col_major(2, 3) << "\n\n";
}

__global__ void vector_add_kernel(
    float const* a, float const* b, float* c, int rows, int cols) {
  auto layout = make_layout(make_shape(rows, cols), LayoutRight{});
  Tensor tA = make_tensor(make_gmem_ptr(a), layout);
  Tensor tB = make_tensor(make_gmem_ptr(b), layout);
  Tensor tC = make_tensor(make_gmem_ptr(c), layout);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < rows && j < cols) {
    tC(i, j) = tA(i, j) + tB(i, j);
  }
}

int main() {
  demo_layouts();

  constexpr int rows = 4;
  constexpr int cols = 8;
  constexpr int n = rows * cols;

  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_c(n, 0.f);
  std::vector<float> h_ref(n);

  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
    h_ref[i] = h_a[i] + h_b[i];
  }

  device_init(0);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(8, 8);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  if (h_c != h_ref) {
    std::cerr << "Vector add verification failed.\n";
    return 1;
  }

  std::cout << "GPU vector add with CuTe tensors: Passed.\n";
  return 0;
}