#pragma once
// ---------------------------------------------------------------------------
// common.cuh
//
// Inline-PTX helpers for driving Blackwell's 5th-generation Tensor Core
// instructions (tcgen05) and the TMA / mbarrier machinery they depend on.
//
// Requires: sm_100a target, CUDA 12.8+ (tcgen05 does not exist before
// Blackwell, and is a different instruction family from Hopper's wgmma).
//
// The shared-memory matrix descriptor and instruction-descriptor encodings
// below follow the documented, benchmarked, correctness-checked pattern
// from "tcgen05 for dummies" (gau-nernst, Dec 2025), cross-referenced
// against the PTX ISA's tcgen05 sections. That's a substantially more
// solid basis than reconstructing the bit layout from memory, but tcgen05
// is new enough that you should still treat this as a starting point --
// verify against the PTX ISA doc for your CUDA version and against
// CUTLASS's SM100 collectives before trusting numerics in production.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cuda_bf16.h>

__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// ---------------------------------------------------------------------------
// mbarrier: init, arrive/expect-tx, and a spin-wait on phase parity.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint32_t mbar_addr, uint32_t arrival_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(mbar_addr), "r"(arrival_count));
}

// Must be called once after initializing any mbarrier(s), before an async
// (TMA / tcgen05) operation references them, so the async proxy sees them.
__device__ __forceinline__ void fence_mbarrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t mbar_addr, uint32_t num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(mbar_addr), "r"(num_bytes) : "memory");
}

// Busy-waits until mbar's current phase (parity 0/1) completes.
__device__ __forceinline__ void mbar_wait(uint32_t mbar_addr, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, 10000000;\n"
        "@P1 bra DONE;\n"
        "bra WAIT_LOOP;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_addr), "r"(phase));
}

// ---------------------------------------------------------------------------
// TMA: issue a 2D global -> shared bulk-tensor copy, tracked by an mbarrier.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tma_load_2d(uint32_t smem_dst, const void* tensor_map,
                                             int32_t coord_x, int32_t coord_y,
                                             uint32_t mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_dst), "l"(tensor_map), "r"(coord_x), "r"(coord_y), "r"(mbar_addr)
        : "memory");
}

// ---------------------------------------------------------------------------
// Tensor Memory (TMEM) allocation.
//
// TMEM is allocated in units of columns (all 128 rows come along for free).
// tcgen05.alloc writes the resulting TMEM address into a *shared memory*
// destination -- it does not return it directly -- because allocation is a
// cooperative, warp-scope operation.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tcgen05_alloc(uint32_t smem_dst, uint32_t num_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :: "r"(smem_dst), "r"(num_cols));
}

__device__ __forceinline__ void tcgen05_dealloc(uint32_t tmem_addr, uint32_t num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :: "r"(tmem_addr), "r"(num_cols));
}

// Required around tcgen05.mma / tcgen05.ld to establish correct ordering
// with plain (non-async-proxy) shared/tensor-memory accesses.
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

// Signals completion of all tcgen05.mma ops issued so far (by this thread)
// on the given mbarrier -- the tcgen05 analogue of wgmma's commit_group.
__device__ __forceinline__ void tcgen05_commit(uint32_t mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                 :: "r"(mbar_addr) : "memory");
}

// ---------------------------------------------------------------------------
// Shared-memory matrix descriptor (64-bit), K-major layout, no swizzling.
//
// Field packing mirrors the same address/LBO/SBO family used by Hopper's
// wgmma descriptor (bits 0-13 address>>4, bits 16-29 LBO>>4, bits 32-45
// SBO>>4), plus bit 46 which tcgen05 uses as a fixed mode flag for this
// (unswizzled, K-major) layout. See PTX ISA "tcgen05 shared memory
// descriptor" / "tcgen05 leading and stride dimension byte offset".
//
// For unswizzled tcgen05, each 8-row x 16-byte "core matrix" must be a
// contiguous chunk in shared memory (this dictates the buffer layout used
// in tcgen05_gemm.cu):
//   LBO (leading dim byte offset) = row_count_of_this_operand * 16
//   SBO (stride dim byte offset)  = 8 * 16   (one core matrix's byte span)
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t desc_field(uint32_t x) {
    return (uint64_t)((x & 0x3FFFFu) >> 4);
}

__device__ __forceinline__ uint64_t make_tcgen05_desc(uint32_t smem_addr, uint32_t operand_rows) {
    const uint32_t lbo = operand_rows * 16u;
    const uint32_t sbo = 8u * 16u;
    return desc_field(smem_addr)
         | (desc_field(lbo) << 16)
         | (desc_field(sbo) << 32)
         | (1ULL << 46); // unswizzled K-major mode flag
}

// ---------------------------------------------------------------------------
// tcgen05.mma.cta_group::1.kind::f16  (covers fp16 and bf16 inputs)
//
//   d_tmem         : TMEM address holding the fp32 accumulator tile
//   a_desc/b_desc  : shared-memory matrix descriptors for this MMA_K slice
//   idesc          : instruction descriptor (dtypes + MMA_M/MMA_N), see
//                    make_idesc_bf16_f32() in tcgen05_gemm.cu
//   enable_input_d : 0 => D = A@B (overwrite); 1 => D = A@B + D (accumulate)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tcgen05_mma_f16(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                                 uint32_t idesc, uint32_t enable_input_d) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_input_d)
        : "memory");
}
