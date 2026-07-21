# tcgen05 minimal example (Blackwell / SM100a)

A single-CTA BF16→FP32 GEMM (`128xK * Kx128 -> 128x128`, looped over K) that
shows the core tcgen05 plumbing on Blackwell: TMA loads via
`cp.async.bulk.tensor`, Tensor Memory (TMEM) allocation, the
`tcgen05.mma` instruction and its shared-memory descriptors, and reading
the accumulator back out with `tcgen05.ld`.

`tcgen05` replaces Hopper's `wgmma` entirely on Blackwell — it's a
different instruction family (single-thread issue instead of warpgroup,
accumulator lives in a new on-chip Tensor Memory instead of registers, A/B
descriptors instead of register operands). If you're coming from a WGMMA
kernel, don't expect to reuse much beyond the general TMA/mbarrier
producer-consumer shape.

This is intentionally the *simplest* working shape: one threadblock, one
output tile, no swizzling, no pipelining, no warp specialization, no 2-SM
cooperative MMA. All of those are real, documented next steps — see
"Where to go from here" below — and each one roughly doubles throughput
in the reference this was built from.

## Requirements

- An SM100a GPU (B200/GB200-class). tcgen05 does not exist on Hopper
  (H100/H200) or on consumer/workstation Blackwell (SM120) — those need a
  different, MMA-based code path.
- CUDA 12.8 or newer.

## Build & run

```bash
make
./tcgen05_gemm
```

Expected output on success:

```
Max abs diff vs CPU reference: <small number>
PASS
```

## Layout

- `src/common.cuh` — mbarrier init/wait, TMA issue, TMEM alloc/dealloc,
  the `tcgen05.mma` / `tcgen05.commit` / `tcgen05.ld` wrappers, and the
  shared-memory descriptor + instruction-descriptor encoders.
- `src/tcgen05_gemm.cu` — host-side tensor map setup (via the
  `cuTensorMapEncodeTiled` driver entry point) and the kernel itself, plus
  a host driver with a CPU reference check.

## Where this comes from, and what to double-check

Unlike WGMMA, tcgen05 is new enough (PTX ISA 8.4+, late 2025) that I
didn't want to reconstruct its descriptor encoding from memory alone. This
project follows the specific instruction sequence and descriptor layout
documented and *benchmarked* in ["tcgen05 for dummies"](https://gau-nernst.github.io/tcgen05/)
(Dec 2025) — the "basic tcgen05 kernel" section, which the author verified
against a CPU reference on real B200 hardware before optimizing further.
The code here is my own rewrite of that structure into a single
self-contained file, not a copy — so it's still worth validating on real
hardware before you build on top of it. If you hit `FAIL`:

1. Check `$CUDA_HOME/doc/pdf/ptx_isa_*.pdf`, sections "tcgen05 shared
   memory descriptor," "tcgen05 instruction descriptor," and "tcgen05
   canonical layouts," against `common.cuh`'s `make_tcgen05_desc()` and
   `make_idesc_bf16_f32()`.
2. Compare against `cutlass::arch` SM100 collectives in
   [CUTLASS](https://github.com/NVIDIA/cutlass) — the authoritative,
   actively-maintained implementation — or against the source behind the
   blog post above: <https://github.com/gau-nernst/learn-cuda/tree/main/02e_matmul_sm100>.
3. `make ptx` and check the emitted `tcgen05.mma` / `tcgen05.ld` lines
   match what you intend.

## Where to go from here

In roughly the order the source material adds them (each with a real
measured speedup on B200):

1. **128-byte swizzled shared memory** — widens the TMA tile from 16 to
   128 bytes and sets the swizzle bits in the descriptor; ~2.7x over this
   unswizzled version.
2. **Pipelining** — prefetch several K-chunks ahead with multiple
   shared-memory buffers and one mbarrier per stage.
3. **Warp specialization** — dedicate separate warps to issuing TMA vs.
   `tcgen05.mma`, so Tensor Cores don't wait on the issuing thread's other
   work.
4. **2-SM cooperative MMA** (`cta_group::2` + thread block clusters) —
   two adjacent CTAs jointly compute one larger tile.
5. **Persistent kernel with a 4th "epilogue" warp group** — overlap
   `tcgen05.ld`/global-store epilogue of one tile with TMA/MMA of the
   next, and tile over the whole grid instead of one CTA per output tile.

Also worth reading directly: the PTX ISA's tcgen05 sections, NVIDIA's
CUTLASS SM100 GEMM examples, and the Colfax/Modular Blackwell GEMM
write-ups referenced in the blog post above.

## Reference
The code for this project was developed using Claude
