// ---------------------------------------------------------------------------
// tcgen05_gemm.cu
//
// Minimal single-CTA Blackwell tcgen05 example:
//
//     C[128x128] = A[128xK] * B_T[128xK]^T      (bf16 in, fp32 accumulate)
//
// One threadblock (128 threads = 4 warps) computes one 128x128 output tile,
// looping over K in chunks of BLOCK_K. This follows the "basic tcgen05
// kernel" structure (TMA -> tcgen05.mma -> tcgen05.ld epilogue, no
// swizzling, no pipelining, no warp specialization, single SM) documented
// and correctness-checked in gau-nernst's "tcgen05 for dummies" (Dec 2025),
// adapted here into a single self-contained file.
//
// Deliberately left out for simplicity (see that writeup for how to add
// them): 128B-swizzled shared memory (~2.7x faster), load/compute
// pipelining, warp specialization, 2-SM cooperative MMA, and tiling the
// grid over multiple output tiles / SMs.
//
// Build:  make            (requires CUDA 12.8+, targets sm_100a)
// Run:    ./tcgen05_gemm
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include "common.cuh"

constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 256;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 64;
constexpr int MMA_K   = 16; // bf16 elements per MMA K-step (= 32 bytes)
constexpr int TB_SIZE = 128; // 4 warps; tcgen05's "Layout D" epilogue needs >= 4

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_e));                                 \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CU_CHECK(expr)                                                       \
    do {                                                                     \
        CUresult _r = (expr);                                                \
        if (_r != CUDA_SUCCESS) {                                            \
            const char* msg = nullptr;                                      \
            cuGetErrorString(_r, &msg);                                      \
            fprintf(stderr, "CUDA driver error %s:%d: %s\n", __FILE__,       \
                    __LINE__, msg ? msg : "?");                              \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Instruction descriptor for tcgen05.mma.kind::f16 with fp32 accumulation
// and bf16 operands. Field offsets per the PTX ISA "tcgen05 instruction
// descriptor" table.
constexpr uint32_t make_idesc_bf16_f32() {
    return (1u << 4)                        // D dtype = FP32
         | (1u << 7)                        // A dtype = BF16
         | (1u << 10)                       // B dtype = BF16
         | ((uint32_t)(BLOCK_N >> 3) << 17)  // MMA_N
         | ((uint32_t)(BLOCK_M >> 4) << 24); // MMA_M
}

// A: M x K, row-major (K contiguous).
// Bt: N x K, row-major (K contiguous) -- B pre-transposed, same convention
// used for K-major operands in Hopper wgmma examples.
__global__ __launch_bounds__(TB_SIZE) void tcgen05_gemm_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    float* __restrict__ C) {

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    // Dynamic shared memory: BLOCK_K/8 contiguous [BLOCK_M or BLOCK_N, 16B]
    // slices for A followed by B -- the layout tcgen05's unswizzled
    // descriptor requires (each 8-row x 16-byte core matrix contiguous).
    extern __shared__ __align__(128) char smem[];
    const uint32_t A_smem = smem_ptr_to_uint(smem);
    const uint32_t B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);

    __shared__ uint64_t mbar[1];
    __shared__ int tmem_addr_smem[1];
    const uint32_t mbar_addr = smem_ptr_to_uint(mbar);

    if (tid == 0) {
        mbar_init(mbar_addr, 1);
        fence_mbarrier_init();
    } else if (warp_id == 1) {
        // One full warp performs the (cooperative) TMEM allocation.
        const uint32_t alloc_dst = smem_ptr_to_uint(tmem_addr_smem);
        tcgen05_alloc(alloc_dst, BLOCK_N);
    }
    __syncthreads();

    const uint32_t taddr = (uint32_t)tmem_addr_smem[0];
    const uint32_t idesc = make_idesc_bf16_f32();

    int phase = 0;
    constexpr int num_k_iters = K / BLOCK_K;
    constexpr int num_slices = BLOCK_K / 8;   // TMA slices per outer K-iter
    constexpr int num_mma_steps = BLOCK_K / MMA_K;

    for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
        if (tid == 0) {
            for (int s = 0; s < num_slices; s++) {
                const int off_k = iter_k * BLOCK_K + s * 8;
                tma_load_2d(A_smem + s * BLOCK_M * 16, &A_tmap, off_k, /*row=*/0, mbar_addr);
                tma_load_2d(B_smem + s * BLOCK_N * 16, &B_tmap, off_k, /*row=*/0, mbar_addr);
            }
            constexpr uint32_t cp_bytes = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(__nv_bfloat16);
            mbar_arrive_expect_tx(mbar_addr, cp_bytes);
        }
        mbar_wait(mbar_addr, phase);
        phase ^= 1;

        tcgen05_fence_after_thread_sync();

        if (tid == 0) {
            for (int k = 0; k < num_mma_steps; k++) {
                const uint32_t a_addr = A_smem + k * BLOCK_M * 32;
                const uint32_t b_addr = B_smem + k * BLOCK_N * 32;
                const uint64_t a_desc = make_tcgen05_desc(a_addr, BLOCK_M);
                const uint64_t b_desc = make_tcgen05_desc(b_addr, BLOCK_N);
                // Only the very first MMA of the whole K-loop overwrites D;
                // every other step (including later outer iterations)
                // accumulates into the existing TMEM tile.
                const uint32_t enable_d = (iter_k == 0 && k == 0) ? 0u : 1u;
                tcgen05_mma_f16(taddr, a_desc, b_desc, idesc, enable_d);
            }
            tcgen05_commit(mbar_addr);
        }
        mbar_wait(mbar_addr, phase);
        phase ^= 1;
    }

    // --- Epilogue: TMEM -> registers -> global C ---------------------------
    tcgen05_fence_after_thread_sync();

    // With cta_group::1 and MMA_M == TB_SIZE == 128, each thread owns
    // exactly one output row: row = warp_id*32 + lane == tid.
    const int out_row = tid;
    for (int n = 0; n < BLOCK_N / 8; n++) {
        float v[8];
        const uint32_t addr = taddr + ((uint32_t)(warp_id * 32) << 16) + (uint32_t)(n * 8);
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
            : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
              "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
            : "r"(addr));
        tcgen05_wait_ld();

        for (int i = 0; i < 8; i++) C[out_row * N + n * 8 + i] = v[i];
    }

    __syncthreads();
    if (warp_id == 0) tcgen05_dealloc(taddr, BLOCK_N);
}

// Encodes a 2D tensor map for an (rows x K) bf16 matrix, sliced into
// [8, rows] boxes of 16 bytes each -- the layout tcgen05's unswizzled
// descriptor requires. See common.cuh / README for why.
static void init_2d_tmap(CUtensorMap* tmap, const __nv_bfloat16* ptr,
                          uint64_t num_rows, uint64_t num_cols_k) {
    static PFN_cuTensorMapEncodeTiled cuTensorMapEncodeTiled_fn = nullptr;
    if (!cuTensorMapEncodeTiled_fn) {
        void* fn = nullptr;
        CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault));
        cuTensorMapEncodeTiled_fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled>(fn);
    }

    constexpr uint32_t rank = 2;
    uint64_t globalDim[rank] = {8, num_rows}; // 8 bf16 elements = 16 bytes
    uint64_t globalStrides[rank - 1] = {num_cols_k * sizeof(__nv_bfloat16)};
    uint32_t boxDim[rank] = {8, (uint32_t)BLOCK_M}; // BLOCK_M == BLOCK_N here
    uint32_t elementStrides[rank] = {1, 1};

    CU_CHECK(cuTensorMapEncodeTiled_fn(
        tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void*)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main() {
    std::vector<__nv_bfloat16> hA(M * K), hBt(N * K);
    std::vector<float> hC(M * N, 0.0f), hC_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) hA[i] = __float2bfloat16((float)((i % 7) - 3) * 0.5f);
    for (int i = 0; i < N * K; ++i) hBt[i] = __float2bfloat16((float)((i % 5) - 2) * 0.5f);

    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += __bfloat162float(hA[m * K + k]) * __bfloat162float(hBt[n * K + k]);
            hC_ref[m * N + n] = acc;
        }

    __nv_bfloat16 *dA, *dBt;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dBt, N * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBt, hBt.data(), N * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));

    CUDA_CHECK(cudaFree(0)); // force context init before driver-API tensor map calls
    CUtensorMap A_tmap, B_tmap;
    init_2d_tmap(&A_tmap, dA, M, K);
    init_2d_tmap(&B_tmap, dBt, N, K);

    const size_t smem_size = (size_t)(BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaFuncSetAttribute(tcgen05_gemm_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_size));

    tcgen05_gemm_kernel<<<1, TB_SIZE, smem_size>>>(A_tmap, B_tmap, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_diff = 0.0;
    for (int i = 0; i < M * N; ++i)
        max_abs_diff = std::max(max_abs_diff, (double)std::fabs(hC[i] - hC_ref[i]));

    printf("Max abs diff vs CPU reference: %g\n", max_abs_diff);
    printf(max_abs_diff < 1.0 ? "PASS\n" : "FAIL (check descriptors / tensor maps in this file)\n");

    cudaFree(dA);
    cudaFree(dBt);
    cudaFree(dC);
    return 0;
}
