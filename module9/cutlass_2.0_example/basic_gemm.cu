// basic_gemm.cu
//
// Minimal CUTLASS 2.0 example: computes C = alpha * A * B + beta * C
// using a single-precision GEMM on the device, then checks the result
// against a naive host reference implementation.
//
// Build:
//   mkdir build && cd build
//   cmake .. -DCUTLASS_DIR=/path/to/cutlass
//   make
//   ./basic_gemm [M N K]

#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t status_ = (call);                                          \
    if (status_ != cudaSuccess) {                                          \
      std::cerr << "CUDA error " << cudaGetErrorString(status_)            \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                       \
  } while (0)

// Naive column-major reference GEMM, used only to check correctness.
void ReferenceGemm(int M, int N, int K,
                    float alpha,
                    std::vector<float> const &A,
                    std::vector<float> const &B,
                    float beta,
                    std::vector<float> &C) {
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      float accumulator = 0.0f;
      for (int k = 0; k < K; ++k) {
        accumulator += A[m + k * M] * B[k + n * K];
      }
      C[m + n * M] = alpha * accumulator + beta * C[m + n * M];
    }
  }
}

// Wraps a CUTLASS 2.0 device-level SGEMM: all matrices column-major, no transposes.
cudaError_t CutlassSgemmNN(int M, int N, int K,
                            float alpha,
                            float const *A, int lda,
                            float const *B, int ldb,
                            float beta,
                            float *C, int ldc) {
  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<
      float, ColumnMajor,   // element/layout of A
      float, ColumnMajor,   // element/layout of B
      float, ColumnMajor>;  // element/layout of C

  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args(
      {M, N, K},
      {A, lda},
      {B, ldb},
      {C, ldc},
      {C, ldc},
      {alpha, beta});

  cutlass::Status status = gemm_operator(args);

  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

int main(int argc, char const *argv[]) {
  int M = 512;
  int N = 256;
  int K = 128;

  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }

  float const alpha = 1.0f;
  float const beta = 0.0f;

  int const lda = M;
  int const ldb = K;
  int const ldc = M;

  size_t const sizeA = static_cast<size_t>(lda) * K;
  size_t const sizeB = static_cast<size_t>(ldb) * N;
  size_t const sizeC = static_cast<size_t>(ldc) * N;

  std::vector<float> hostA(sizeA);
  std::vector<float> hostB(sizeB);
  std::vector<float> hostC(sizeC, 0.0f);
  std::vector<float> hostCRef(sizeC, 0.0f);

  std::srand(2024);
  for (auto &v : hostA) v = static_cast<float>((std::rand() % 9) - 4);
  for (auto &v : hostB) v = static_cast<float>((std::rand() % 9) - 4);

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceC = nullptr;

  CUDA_CHECK(cudaMalloc(&deviceA, sizeA * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceB, sizeB * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceC, sizeC * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(deviceA, hostA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceB, hostB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceC, hostC.data(), sizeC * sizeof(float), cudaMemcpyHostToDevice));

  cudaError_t result = CutlassSgemmNN(M, N, K, alpha, deviceA, lda, deviceB, ldb, beta, deviceC, ldc);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM launch failed" << std::endl;
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hostC.data(), deviceC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

  ReferenceGemm(M, N, K, alpha, hostA, hostB, beta, hostCRef);

  bool passed = true;
  for (size_t i = 0; i < sizeC; ++i) {
    if (std::abs(hostC[i] - hostCRef[i]) > 1e-3f) {
      passed = false;
      break;
    }
  }

  std::cout << "M=" << M << " N=" << N << " K=" << K << " : "
            << (passed ? "PASSED" : "FAILED") << std::endl;

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
