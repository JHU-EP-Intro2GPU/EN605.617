/*
 * CuTe SGEMM tutorial example (simplified from CUTLASS examples/cute/tutorial/sgemm_1.cu).
 *
 * Computes C = A * B^T using CuTe layouts, tensors, copy, and gemm.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include "cute_helper.h"

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
                 TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
                 TC* C, CStride dC, CSmemLayout, CThreadLayout tC,
                 Alpha alpha, Beta beta) {
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);
  static_assert(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tB));
  CUTE_STATIC_ASSERT_V(size(tC) == size(tA));

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  Tensor tAgA = local_partition(gA, tA, threadIdx.x);
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);
  Tensor tBgB = local_partition(gB, tB, threadIdx.x);
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
  Tensor tCrC = make_tensor_like(tCgC);
  clear(tCrC);

  auto K_TILE_MAX = size<2>(tAgA);
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    copy(tAgA(_, _, k_tile), tAsA);
    copy(tBgB(_, _, k_tile), tBsB);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    gemm(tCsA, tCsB, tCrC);
    __syncthreads();
  }

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k,
             Alpha alpha,
             TA const* A, int ldA,
             TB const* B, int ldB,
             Beta beta,
             TC* C, int ldC,
             cudaStream_t stream = 0) {
  using namespace cute;

  auto prob_shape = make_shape(int(m), int(n), int(k));
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = make_layout(make_shape(bM, bK));
  auto sB = make_layout(make_shape(bN, bK));
  auto sC = make_layout(make_shape(bM, bN));
 
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(int(m), bM)), size(ceil_div(int(n), bN)));

  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
      prob_shape, cta_tiler,
      A, dA, sA, tA,
      B, dB, sB, tB,
      C, dC, sC, tC,
      alpha, beta);
}

void reference_gemm_nt(int m, int n, int k,
                       float alpha,
                       float const* A, int ldA,
                       float const* B, int ldB,
                       float beta,
                       float* C, int ldC) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      float acc = 0.f;
      for (int kk = 0; kk < k; ++kk) {
        acc += A[i + kk * ldA] * B[j + kk * ldB];
      }
      C[i + j * ldC] = alpha * acc + beta * C[i + j * ldC];
    }
  }
}

int main(int argc, char** argv) {
  int m = 256;
  int n = 256;
  int k = 256;
  if (argc >= 2) sscanf(argv[1], "%d", &m);
  if (argc >= 3) sscanf(argv[2], "%d", &n);
  if (argc >= 4) sscanf(argv[3], "%d", &k);

  float alpha = 1.f;
  float beta = 0.f;

  std::cout << "CuTe SGEMM: C = A * B^T\n";
  std::cout << "M=" << m << " N=" << n << " K=" << k << "\n";

  device_init(0);

  thrust::host_vector<float> h_A(m * k);
  thrust::host_vector<float> h_B(n * k);
  thrust::host_vector<float> h_C(m * n, 0.f);
  thrust::host_vector<float> h_ref(m * n, 0.f);

  for (int j = 0; j < m * k; ++j) {
    h_A[j] = static_cast<float>(2 * (rand() / double(RAND_MAX)) - 1);
  }

  for (int j = 0; j < n * k; ++j) {
    h_B[j] = static_cast<float>(2 * (rand() / double(RAND_MAX)) - 1);
  }

  reference_gemm_nt(m, n, k, alpha, h_A.data(), m, h_B.data(), n, beta, h_ref.data(), m);

  thrust::device_vector<float> d_A = h_A;
  thrust::device_vector<float> d_B = h_B;
  thrust::device_vector<float> d_C(m * n, 0.f);

  gemm_nt(m, n, k, alpha,
          d_A.data().get(), m,
          d_B.data().get(), n,
          beta,
          d_C.data().get(), m);
  CUDA_CHECK_LAST();

  thrust::host_vector<float> h_result = d_C;

  const float tolerance = 1e-3f;
  for (int i = 0; i < m * n; ++i) {
    if (std::abs(h_result[i] - h_ref[i]) > tolerance) {
      std::cerr << "Verification failed at index " << i
                << ": got " << h_result[i] << ", expected " << h_ref[i] << "\n";
      return 1;
    }
  }

  std::cout << "Passed.\n";
  return 0;
}
