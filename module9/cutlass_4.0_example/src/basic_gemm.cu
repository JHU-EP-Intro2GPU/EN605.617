/*
 * Minimal CUTLASS 4.0 example: single-precision GEMM (SGEMM).
 *
 * Based on NVIDIA CUTLASS examples/00_basic_gemm, adapted for a standalone project.
 */

#include <iostream>
#include <sstream>
#include <vector>

#include "helper.h"
#include "cutlass/gemm/device/gemm.h"

cudaError_t CutlassSgemmNN(
    int M, int N, int K,
    float alpha,
    float const* A, int lda,
    float const* B, int ldb,
    float beta,
    float* C, int ldc) {

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using CutlassGemm = cutlass::gemm::device::Gemm<
      float, ColumnMajor,
      float, ColumnMajor,
      float, ColumnMajor>;

  CutlassGemm gemm_operator;
  CutlassGemm::Arguments args(
      {M, N, K},
      {A, lda},
      {B, ldb},
      {C, ldc},
      {C, ldc},
      {alpha, beta});

  cutlass::Status status = gemm_operator(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

__global__ void InitializeMatrix_kernel(
    float* matrix, int rows, int columns, int seed = 0) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);
    matrix[offset] = value;
  }
}

cudaError_t InitializeMatrix(float* matrix, int rows, int columns, int seed = 0) {
  dim3 block(16, 16);
  dim3 grid(
      (rows + block.x - 1) / block.x,
      (columns + block.y - 1) / block.y);

  InitializeMatrix_kernel<<<grid, block>>>(matrix, rows, columns, seed);
  return cudaGetLastError();
}

cudaError_t AllocateMatrix(float** matrix, int rows, int columns, int seed = 0) {
  size_t sizeof_matrix = sizeof(float) * rows * columns;

  cudaError_t result = cudaMalloc(reinterpret_cast<void**>(matrix), sizeof_matrix);
  if (result != cudaSuccess) {
    return result;
  }

  result = cudaMemset(*matrix, 0, sizeof_matrix);
  if (result != cudaSuccess) {
    return result;
  }

  return InitializeMatrix(*matrix, rows, columns, seed);
}

__global__ void ReferenceGemm_kernel(
    int M, int N, int K,
    float alpha,
    float const* A, int lda,
    float const* B, int ldb,
    float beta,
    float* C, int ldc) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;
    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }
    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

cudaError_t ReferenceGemm(
    int M, int N, int K,
    float alpha,
    float const* A, int lda,
    float const* B, int ldb,
    float beta,
    float* C, int ldc) {
  dim3 block(16, 16);
  dim3 grid(
      (M + block.x - 1) / block.x,
      (N + block.y - 1) / block.y);

  ReferenceGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}

cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta) {
  int lda = M;
  int ldb = K;
  int ldc = M;
  size_t sizeof_C = sizeof(float) * ldc * N;

  float *A = nullptr, *B = nullptr, *C_cutlass = nullptr, *C_reference = nullptr;

  cudaError_t result = AllocateMatrix(&A, M, K, 0);
  if (result != cudaSuccess) return result;

  result = AllocateMatrix(&B, K, N, 17);
  if (result != cudaSuccess) { cudaFree(A); return result; }

  result = AllocateMatrix(&C_cutlass, M, N, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); return result; }

  result = AllocateMatrix(&C_reference, M, N, 101);
  if (result != cudaSuccess) {
    cudaFree(A); cudaFree(B); cudaFree(C_cutlass);
    return result;
  }

  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);
  if (result != cudaSuccess) {
    cudaFree(A); cudaFree(B); cudaFree(C_cutlass); cudaFree(C_reference);
    return result;
  }

  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);
  if (result != cudaSuccess) {
    cudaFree(A); cudaFree(B); cudaFree(C_cutlass); cudaFree(C_reference);
    return result;
  }

  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
  if (result != cudaSuccess) {
    cudaFree(A); cudaFree(B); cudaFree(C_cutlass); cudaFree(C_reference);
    return result;
  }

  std::vector<float> host_cutlass(ldc * N, 0);
  std::vector<float> host_reference(ldc * N, 0);

  result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    cudaFree(A); cudaFree(B); cudaFree(C_cutlass); cudaFree(C_reference);
    return result;
  }

  result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    cudaFree(A); cudaFree(B); cudaFree(C_cutlass); cudaFree(C_reference);
    return result;
  }

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  if (host_cutlass != host_reference) {
    std::cerr << "CUTLASS results incorrect." << std::endl;
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

int main(int argc, const char* argv[]) {
  int problem[3] = {128, 128, 128};
  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(argv[i]);
    ss >> problem[i - 1];
  }

  float scalars[2] = {1.f, 0.f};
  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(argv[i]);
    ss >> scalars[i - 4];
  }

  cudaError_t result = TestCutlassGemm(
      problem[0], problem[1], problem[2], scalars[0], scalars[1]);

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  } else {
    std::cerr << "Failed: " << cudaGetErrorString(result) << std::endl;
  }

  return result == cudaSuccess ? 0 : -1;
}
