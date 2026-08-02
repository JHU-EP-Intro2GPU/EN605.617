// basic_gemm_3x.cu
//
// Minimal CUTLASS 3.0 example targeting Hopper (sm90a).
//
// Computes D = alpha * A * B + beta * C in FP16, with FP32 accumulation,
// using the CUTLASS 3.x "CollectiveBuilder" API: the mainloop and
// epilogue are each auto-selected from tile shape + cluster shape +
// architecture, then assembled into a GemmUniversalAdapter. This is
// the 3.x replacement for hand-picking a `cutlass::gemm::device::Gemm<...>`
// template as in CUTLASS 2.x.
//
// Build:
//   mkdir build && cd build
//   cmake .. -DCUTLASS_DIR=/path/to/cutlass
//   make
//   ./basic_gemm_3x [M N K]
//
// Requires: CUDA 12.0+, an sm90 (Hopper) GPU, CUTLASS 3.x headers.

#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t status_ = (call);                                          \
    if (status_ != cudaSuccess) {                                          \
      std::cerr << "CUDA error " << cudaGetErrorString(status_)            \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                       \
  } while (0)

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// Tile shape per threadblock cluster iteration, and the cluster shape
// itself. These are the two knobs CUTLASS 3.x asks for up front; the
// builders below pick everything else (stage count, schedule, swizzling).
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, 8,
    ElementC, LayoutC, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,  // M, N, K, batch
    CollectiveMainloop,
    CollectiveEpilogue
  >;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// Naive host reference GEMM (row-major A, column-major B, row-major C),
// computed in fp32 regardless of the device element types.
void ReferenceGemm(int M, int N, int K,
                    float alpha,
                    std::vector<cutlass::half_t> const &A,
                    std::vector<cutlass::half_t> const &B,
                    float beta,
                    std::vector<cutlass::half_t> &C) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float accumulator = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a = static_cast<float>(A[m * K + k]);   // row-major
        float b = static_cast<float>(B[k + n * K]);   // column-major
        accumulator += a * b;
      }
      float c = static_cast<float>(C[m * N + n]);
      C[m * N + n] = static_cast<cutlass::half_t>(alpha * accumulator + beta * c);
    }
  }
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

  std::vector<cutlass::half_t> hostA(static_cast<size_t>(M) * K);
  std::vector<cutlass::half_t> hostB(static_cast<size_t>(K) * N);
  std::vector<cutlass::half_t> hostC(static_cast<size_t>(M) * N, cutlass::half_t(0));
  std::vector<cutlass::half_t> hostRef(static_cast<size_t>(M) * N, cutlass::half_t(0));

  std::srand(2024);
  for (auto &v : hostA) v = static_cast<cutlass::half_t>((std::rand() % 9) - 4);
  for (auto &v : hostB) v = static_cast<cutlass::half_t>((std::rand() % 9) - 4);

  cutlass::half_t *deviceA = nullptr;
  cutlass::half_t *deviceB = nullptr;
  cutlass::half_t *deviceC = nullptr;
  cutlass::half_t *deviceD = nullptr;

  CUDA_CHECK(cudaMalloc(&deviceA, hostA.size() * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&deviceB, hostB.size() * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&deviceC, hostC.size() * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&deviceD, hostC.size() * sizeof(cutlass::half_t)));

  CUDA_CHECK(cudaMemcpy(deviceA, hostA.data(), hostA.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceB, hostB.data(), hostB.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceC, hostC.data(), hostC.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));

  StrideA strideA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB strideB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC strideC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD strideD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {deviceA, strideA, deviceB, strideB},
      {{alpha, beta}, deviceC, strideC, deviceD, strideD}
  };

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }

  if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
    std::cerr << "This problem size / configuration is not supported on this GPU." << std::endl;
    return EXIT_FAILURE;
  }

  if (gemm_op.initialize(arguments, workspace) != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize CUTLASS GEMM." << std::endl;
    return EXIT_FAILURE;
  }
  if (gemm_op.run() != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM run failed." << std::endl;
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hostC.data(), deviceD, hostC.size() * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));

  ReferenceGemm(M, N, K, alpha, hostA, hostB, beta, hostRef);

  bool passed = true;
  for (size_t i = 0; i < hostC.size(); ++i) {
    float diff = std::abs(static_cast<float>(hostC[i]) - static_cast<float>(hostRef[i]));
    if (diff > 0.5f) {  // loose tolerance: fp16 storage + tensor-core accumulation
      passed = false;
      break;
    }
  }

  std::cout << "M=" << M << " N=" << N << " K=" << K << " : "
            << (passed ? "PASSED" : "FAILED") << std::endl;

  if (workspace) cudaFree(workspace);
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  cudaFree(deviceD);

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
