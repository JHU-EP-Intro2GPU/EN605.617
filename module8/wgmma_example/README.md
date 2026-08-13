# WGMMA minimal example (Hopper / SM90a)

A single-tile FP16→FP32 GEMM (`64x16 * 16x64 -> 64x64`) that shows the raw
plumbing behind Hopper's warpgroup-level MMA instruction: shared-memory
matrix descriptors, `wgmma.fence` / `wgmma.mma_async` / `wgmma.commit_group`
/ `wgmma.wait_group`, and the accumulator-fragment layout.

It's deliberately not a tuned kernel — no TMA, no pipelining, no tiling
over M/N/K, no swizzled shared memory. The goal is to make the WGMMA call
itself legible before you go build (or go read) something like CUTLASS's
Hopper GEMM collectives.

## Requirements

- An SM90a GPU (H100 or H200). WGMMA does not exist on anything earlier,
  including A100.
- CUDA 12.3 or newer (`nvcc --version`).

## Build & run

```bash
make
./wgmma_gemm
```

Expected output on success:

```
Max abs diff vs CPU reference: <small number>
PASS
```

`make ptx` dumps the generated PTX if you want to eyeball the emitted
`wgmma.mma_async` instruction and descriptor construction.

## Layout

- `src/common.cuh` — descriptor encoding, fence/commit/wait wrappers, the
  `wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16` inline-asm call.
- `src/wgmma_gemm.cu` — kernel (shared-memory load, descriptor setup,
  fragment scatter into `C`) plus a host driver with a CPU reference check.

## The part to double-check before you trust this

The 64-bit shared-memory matrix descriptor's bit layout (start address,
leading/stride byte offsets, swizzle mode) is one of the least-documented
corners of the PTX ISA, and NVIDIA has adjusted the exact packing across
CUDA releases. `common.cuh` implements it as carefully as I could
reconstruct from the PTX ISA docs, and the host program's CPU-reference
check exists specifically so you'll know immediately (via `PASS`/`FAIL`)
if the encoding is off for your toolkit version.

If you hit `FAIL`:

1. Check `$CUDA_HOME/doc/pdf/ptx_isa_*.pdf`, section on asynchronous
   warpgroup-level MMA, for the descriptor bit layout matching your CUDA
   version.
2. Compare against `cutlass::arch::GmmaDescriptor` in the
   [CUTLASS](https://github.com/NVIDIA/cutlass) source — it's the
   authoritative, actively-maintained implementation.
3. `make ptx` and check that the immediates in the emitted
   `wgmma.mma_async` line (scale-d, scale-a, scale-b, trans-a, trans-b)
   match what you intend — they're compile-time constants baked into the
   instruction, not runtime registers.

## Where to go from here

- Swap the plain shared-memory loads for `cp.async.bulk` (TMA) descriptors
  and an `mbarrier`-based wait, which is how real Hopper kernels overlap
  loads with compute.
- Tile over K and issue multiple `wgmma.mma_async` calls between one
  `fence`/`commit_group`/`wait_group` to accumulate across K-chunks
  (drop `scale-d` to `1` after the first call).
- Tile over M/N and use multiple warpgroups per threadblock.
- Read NVIDIA's CUTLASS Hopper GEMM examples (`examples/48_hopper_warp_specialized_gemm`
  and friends) — that's the production version of everything sketched here.


## Reference
The code for this project was developed using Claude