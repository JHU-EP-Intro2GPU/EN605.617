# WMMA minimal example (Volta and later, sm_70+)

A tiled FP16→FP32 GEMM (`M=N=K=256`) using the `nvcuda::wmma` C++ API from
`<mma.h>`. Each warp owns one 16×16×16 MMA tile; a block of warps covers a
larger output tile, and the grid covers the rest of `C`.

## Why WMMA, and when to reach for something else

WMMA is the stable, portable tensor-core API: same C++ interface since
Volta, no raw PTX, no hand-built shared-memory descriptors, compiler
handles fragment register allocation. That portability is also its
ceiling — on Hopper and Blackwell it doesn't use `wgmma`/`tcgen05` under
the hood, so it leaves real throughput on the table on those GPUs. Rule of
thumb:

- **Prototyping, portability across GPU generations, or a first tensor-core
  kernel** → WMMA.
- **Squeezing out Hopper's peak throughput** → `wgmma` (see the companion
  `wgmma_example` project).
- **Squeezing out Blackwell's peak throughput** → `tcgen05` (see the
  companion `tcgen05_example` project).

## Requirements

- Any GPU with compute capability 7.0+ (Volta, Turing, Ampere, Ada,
  Hopper, Blackwell all work — WMMA is architecture-generic).
- CUDA 9.0 or newer (any reasonably current toolkit is fine).

## Build & run

```bash
make                  # targets sm_80 by default
make ARCH=sm_75 run   # override for your GPU, e.g. Turing
```

Expected output on success:

```
Max abs diff vs CPU reference: <small number>
PASS
```

## Layout

- `src/wmma_gemm.cu` — the kernel (fragment load/mma/store loop over K) and
  a host driver with a CPU reference check. No separate header needed —
  WMMA doesn't require the descriptor/mbarrier plumbing WGMMA and tcgen05
  do.

## Notes on the memory layout

`B` is supplied pre-transposed, stored as `Bt` with shape `(N, K)`
row-major, so the B fragment can be loaded `col_major` with leading
dimension `K` straight out of it — no separate transpose step. This is the
same `A: (M,K)` / `Bt: (N,K)` convention used in the companion WGMMA and
tcgen05 example projects, so it's easy to compare the three side by side.

## Where to go from here

- Stage `A`/`B` tiles through shared memory once per block and reuse them
  across all warps in that block, instead of every warp re-reading from
  global memory — the standard next optimization.
- Tile over K with `cp.async` (Ampere+) to overlap loads with compute.
- Try `wmma::load_matrix_sync`/`store_matrix_sync` with different fragment
  layouts (`mem_col_major`, other dtypes like `__nv_bfloat16` or
  `precision::tf32`) — WMMA supports several without changing the overall
  structure.
- Compare against the `wgmma_example` and `tcgen05_example` projects to see
  how the same GEMM shape is expressed on newer hardware.
