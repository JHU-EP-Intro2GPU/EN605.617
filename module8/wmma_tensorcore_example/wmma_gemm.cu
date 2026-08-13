// ---------------------------------------------------------------------------
// wmma_gemm.cu
//
// A tiled FP16 -> FP32 GEMM using the nvcuda::wmma C++ API (mma.h):
//
//     C[MxN] = A[MxK] * B_T[NxK]^T
//
// Unlike WGMMA (Hopper) or tcgen05 (Blackwell), WMMA is a stable, portable,
// fully-documented API -- it works the same way from Volta (sm_70) through
// current architectures, needs no hand-built shared-memory descriptors or
// raw PTX, and the compiler handles register allocation for you. It's also
// the slowest of the three on any given architecture, since newer GPUs have
// faster, lower-level tensor-core paths (wgmma, tcgen05) that WMMA doesn't
// use -- but it's by far the easiest correct starting point.
//
// Each warp computes one WMMA_M x WMMA_N tile of C, looping over K in
// WMMA_K-sized chunks. A tile of warps within a threadblock covers a larger
// output tile; the grid covers the rest of C. Fragments are loaded directly
// from global memory (no shared-memory staging) to keep this readable --
// real kernels usually stage tiles through shared memory for reuse across
// warps in a block.
//
// Build:  make            (default targets sm_80; override with ARCH=sm_70
//                           etc. -- WMMA works on any sm_70+ GPU)
// Run:    ./wmma_gemm
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Problem size (kept as multiples of the WMMA tile for a clean example;
// the kernel itself has boundary checks and works for arbitrary sizes).
constexpr int M = 256;
constexpr int N = 256;
constexpr int K = 256;

// Warps per block, arranged as a 2D grid of tiles: BLOCK_WARPS_M in the
// M direction (each contributes blockDim.x/32 warps) x BLOCK_WARPS_N in
// the N direction (blockDim.y warps). Here: 2 x 4 = 8 warps/block, each
// covering a 16x16 tile of C, so one block computes a 32x64 tile.
constexpr int BLOCK_WARPS_M = 2;
constexpr int BLOCK_WARPS_N = 4;

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_e));                                 \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// A: M x K, row-major.
// Bt: N x K, row-major -- i.e. B pre-transposed. This lets the B fragment
// be loaded as col_major with leading dimension K directly from Bt, which
// is the standard way to feed WMMA a logically K x N operand without a
// separate transpose step.
__global__ void wmma_gemm_kernel(const half* __restrict__ A,
                                  const half* __restrict__ Bt,
                                  float* __restrict__ C) {
    // Which warp within the whole grid, and which output tile it owns.
    const int warp_id_in_block = threadIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    const int warp_m = blockIdx.x * BLOCK_WARPS_M + (warp_id_in_block % BLOCK_WARPS_M);
    const int warp_n = blockIdx.y * BLOCK_WARPS_N + (warp_id_in_block / BLOCK_WARPS_M);

    const int c_row = warp_m * WMMA_M;
    const int c_col = warp_n * WMMA_N;
    if (c_row >= M || c_col >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        // A tile: rows [c_row, c_row+16), cols [k, k+16), row-major, ld = K.
        wmma::load_matrix_sync(a_frag, A + c_row * K + k, K);
        // B tile (via Bt): logical rows [k, k+16) of B, cols [c_col, c_col+16),
        // read col-major with ld = K directly out of the (N, K) Bt buffer.
        wmma::load_matrix_sync(b_frag, Bt + c_col * K + k, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(C + c_row * N + c_col, acc_frag, N, wmma::mem_row_major);
}

int main() {
    std::vector<half> hA(M * K), hBt(N * K);
    std::vector<float> hC(M * N, 0.0f), hC_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) hA[i] = __float2half((float)((i % 7) - 3) * 0.5f);
    for (int i = 0; i < N * K; ++i) hBt[i] = __float2half((float)((i % 5) - 2) * 0.5f);

    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += __half2float(hA[m * K + k]) * __half2float(hBt[n * K + k]);
            hC_ref[m * N + n] = acc;
        }

    half *dA, *dBt;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dBt, N * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBt, hBt.data(), N * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));

    // Each block: BLOCK_WARPS_M x BLOCK_WARPS_N warps -> blockDim.x carries
    // BLOCK_WARPS_M warps' worth of threads (32 each), blockDim.y = BLOCK_WARPS_N.
    dim3 blockDim(BLOCK_WARPS_M * 32, BLOCK_WARPS_N);
    dim3 gridDim((M + (BLOCK_WARPS_M * WMMA_M) - 1) / (BLOCK_WARPS_M * WMMA_M),
                 (N + (BLOCK_WARPS_N * WMMA_N) - 1) / (BLOCK_WARPS_N * WMMA_N));

    wmma_gemm_kernel<<<gridDim, blockDim>>>(dA, dBt, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_diff = 0.0;
    for (int i = 0; i < M * N; ++i)
        max_abs_diff = std::max(max_abs_diff, (double)std::fabs(hC[i] - hC_ref[i]));

    printf("Max abs diff vs CPU reference: %g\n", max_abs_diff);
    printf(max_abs_diff < 1e-2 ? "PASS\n" : "FAIL\n");

    cudaFree(dA);
    cudaFree(dBt);
    cudaFree(dC);
    return 0;
}
