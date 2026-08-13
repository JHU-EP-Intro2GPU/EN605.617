// ---------------------------------------------------------------------------
// wgmma_gemm.cu
//
// Minimal single-tile Hopper WGMMA example: one threadblock (= one
// warpgroup, 128 threads) computes
//
//     C[64x64] = A[64x16] * B_T[64x16]^T          (fp16 in, fp32 accumulate)
//
// i.e. a single m64n64k16 wgmma.mma_async call, no tiling/pipelining, no
// TMA. B is supplied pre-transposed (stored as N x K, row-major) because
// this kernel uses trans-b = 0 ("K-major") for both operands -- see
// common.cuh for what that means for the descriptors.
//
// This is meant as a readable skeleton for the WGMMA plumbing (descriptors,
// fence/commit/wait, the accumulator fragment layout), not a tuned kernel.
// Real GEMMs additionally pipeline TMA loads against multiple wgmma groups,
// tile over M/N/K, and use swizzled shared-memory layouts for throughput.
//
// Build:  make            (requires CUDA 12.3+, targets sm_90a)
// Run:    ./wgmma_gemm
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include "common.cuh"

constexpr int M = 64;
constexpr int N = 64;
constexpr int K = 16;

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_err));                               \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// A: M x K, row-major (K contiguous).
// Bt: N x K, row-major (K contiguous) -- this is B, pre-transposed.
__global__ void wgmma_gemm_kernel(const half* __restrict__ A,
                                   const half* __restrict__ Bt,
                                   float* __restrict__ C) {
    // 128-byte alignment gives the >>4 descriptor encoding room to work with.
    __shared__ __align__(128) half A_smem[M * K];
    __shared__ __align__(128) half B_smem[N * K];

    const int tid = threadIdx.x; // 0..127

    // Cooperative load: 128 threads, 1024 elements each buffer -> 8 per thread.
    for (int i = tid; i < M * K; i += blockDim.x) A_smem[i] = A[i];
    for (int i = tid; i < N * K; i += blockDim.x) B_smem[i] = Bt[i];
    __syncthreads();

    // --- Build shared-memory matrix descriptors --------------------------
    // K-major, no swizzle. Core matrix depth for fp16 is 8 elements of K,
    // so K=16 is two 8-element chunks back to back in shared memory.
    const uint32_t a_base = smem_ptr_to_uint(A_smem);
    const uint32_t b_base = smem_ptr_to_uint(B_smem);

    const uint32_t a_lbo = 8 * K * sizeof(half); // stride between row-groups of 8
    const uint32_t a_sbo = 8 * sizeof(half);     // stride between the two K-chunks
    const uint32_t b_lbo = 8 * K * sizeof(half);
    const uint32_t b_sbo = 8 * sizeof(half);

    uint64_t desc_a = make_wgmma_desc(a_base, a_lbo, a_sbo);
    uint64_t desc_b = make_wgmma_desc(b_base, b_lbo, b_sbo);

    // --- Issue the WGMMA -------------------------------------------------
    float d[32];
    wgmma_fence();
    wgmma_m64n64k16_f16f16f32(d, desc_a, desc_b);
    wgmma_commit_group();
    wgmma_wait_group0();

    // --- Scatter this thread's fragment into C ----------------------------
    // Accumulator fragment layout for m64nNk16 (fp32), one warpgroup = 4
    // warps of 32 lanes, each warp owning 16 consecutive output rows:
    //
    //   warp_id = tid / 32, lane = tid % 32
    //   row0 = warp_id*16 + lane/4,  row1 = row0 + 8
    //   for each 8-wide column group j in [0, N/8):
    //     col0 = j*8 + (lane%4)*2, col1 = col0 + 1
    //     d[4j+0] -> (row0, col0)   d[4j+1] -> (row0, col1)
    //     d[4j+2] -> (row1, col0)   d[4j+3] -> (row1, col1)
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int row0 = warp_id * 16 + lane / 4;
    const int row1 = row0 + 8;

    for (int j = 0; j < N / 8; ++j) {
        const int col0 = j * 8 + (lane % 4) * 2;
        const int col1 = col0 + 1;
        C[row0 * N + col0] = d[4 * j + 0];
        C[row0 * N + col1] = d[4 * j + 1];
        C[row1 * N + col0] = d[4 * j + 2];
        C[row1 * N + col1] = d[4 * j + 3];
    }
}

int main() {
    std::vector<half> hA(M * K), hBt(N * K);
    std::vector<float> hC(M * N, 0.0f), hC_ref(M * N, 0.0f);

    // Small deterministic values so the CPU reference is exact in fp32.
    for (int i = 0; i < M * K; ++i) hA[i] = __float2half((float)((i % 7) - 3) * 0.5f);
    for (int i = 0; i < N * K; ++i) hBt[i] = __float2half((float)((i % 5) - 2) * 0.5f);

    // CPU reference: C[m][n] = sum_k A[m][k] * B[n][k]   (Bt is N x K)
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += __half2float(hA[m * K + k]) * __half2float(hBt[n * K + k]);
            hC_ref[m * N + n] = acc;
        }

    half *dA, *dBt;
    float *dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dBt, N * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBt, hBt.data(), N * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));

    wgmma_gemm_kernel<<<1, 128>>>(dA, dBt, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_diff = 0.0;
    for (int i = 0; i < M * N; ++i)
        max_abs_diff = std::max(max_abs_diff, (double)std::fabs(hC[i] - hC_ref[i]));

    printf("Max abs diff vs CPU reference: %g\n", max_abs_diff);
    printf(max_abs_diff < 1e-2 ? "PASS\n" : "FAIL (check descriptor encoding in common.cuh)\n");

    cudaFree(dA);
    cudaFree(dBt);
    cudaFree(dC);
    return 0;
}
