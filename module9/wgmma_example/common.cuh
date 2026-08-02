#pragma once
// ---------------------------------------------------------------------------
// common.cuh
//
// Small helpers for driving Hopper's warpgroup-level MMA (WGMMA) instruction
// from inline PTX: shared-memory matrix descriptors, the fence/commit/wait
// synchronization primitives, and the wgmma.mma_async call itself.
//
// Requires: sm_90a target, CUDA 12.3+.
//
// IMPORTANT: the 64-bit "matrix descriptor" bit layout below follows the
// PTX ISA section on "Asynchronous Warpgroup-Level Matrix Multiply-
// Accumulate Instructions" as closely as I can reconstruct it, but this is
// genuinely one of the least stable/most under-documented parts of the ISA
// -- the exact field packing has changed across CUDA releases and differs
// by swizzle mode. Before trusting numerics in anything real, diff this
// against `cutlass::arch::GmmaDescriptor` in the CUTLASS source for the
// CUDA version you're targeting, or the PTX ISA doc shipped with your
// toolkit (`$CUDA_HOME/doc/pdf/ptx_isa_*.pdf`).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cuda_fp16.h>

// Bit layout (as documented):
//   bits  0-13 : start address, encoded as (shared_addr >> 4)
//   bits 16-29 : leading dimension byte offset (LBO), encoded >> 4
//   bits 32-45 : stride dimension byte offset (SBO), encoded >> 4
//   bits 49-51 : matrix base offset (only meaningful with swizzling)
//   bits 62-63 : swizzle mode: 0 = none, 1 = 128B, 2 = 64B, 3 = 32B
__device__ __forceinline__ uint64_t make_wgmma_desc(uint32_t smem_addr,
                                                      uint32_t leading_byte_offset,
                                                      uint32_t stride_byte_offset,
                                                      uint8_t swizzle = 0,
                                                      uint8_t base_offset = 0) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr & 0x3FFFFu) >> 4);
    desc |= (uint64_t)((leading_byte_offset & 0x3FFFFu) >> 4) << 16;
    desc |= (uint64_t)((stride_byte_offset  & 0x3FFFFu) >> 4) << 32;
    desc |= (uint64_t)(base_offset & 0x7u) << 49;
    desc |= (uint64_t)(swizzle & 0x3u) << 62;
    return desc;
}

__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Must precede the first wgmma.mma_async in a group.
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

// Closes off a batch of wgmma.mma_async ops as one "group" for wait_group.
__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

// Blocks until at most N wgmma groups are still in flight (N=0 -> drain all).
__device__ __forceinline__ void wgmma_wait_group0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// ---------------------------------------------------------------------------
// wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
//
// One warpgroup (128 threads) computes a 64x64 tile of D (fp32) from a
// 64x16 tile of A (fp16) and a 16x64 tile of B (fp16), using K-major
// (i.e. K-contiguous) shared-memory operands with no swizzling.
//
// Immediates baked into the instruction (must be compile-time constants):
//   scale-d     = 0  -> D is overwritten (not accumulated into)
//   imm-scale-a = 1, imm-scale-b = 1  -> no sign flip
//   trans-a = 0, trans-b = 0          -> both operands read K-major
//
// d[0..31] holds this thread's fragment of the 64x64 output; see
// wgmma_gemm.cu for the register -> (row, col) mapping.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_m64n64k16_f16f16f32(float d[32],
                                                            uint64_t desc_a,
                                                            uint64_t desc_b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 0, 1, 1, 0, 0;\n"
        : "=f"(d[0]),  "=f"(d[1]),  "=f"(d[2]),  "=f"(d[3]),
          "=f"(d[4]),  "=f"(d[5]),  "=f"(d[6]),  "=f"(d[7]),
          "=f"(d[8]),  "=f"(d[9]),  "=f"(d[10]), "=f"(d[11]),
          "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
          "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
          "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
          "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]),
          "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
        : "l"(desc_a), "l"(desc_b)
    );
}
