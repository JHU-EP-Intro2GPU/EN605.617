# CUTLASS 3.0 Basic GEMM Example (Hopper)

A minimal project showing the CUTLASS 3.x device API: `D = alpha*A*B + beta*C`
in FP16 with FP32 accumulation, built using `CollectiveBuilder` for the
mainloop and epilogue, assembled into a `GemmUniversalAdapter`. This is
the 3.x counterpart to hand-picking a `cutlass::gemm::device::Gemm<...>`
template directly, as CUTLASS 2.x does.

## Requirements

- CUDA Toolkit **12.0+** (Hopper's WGMMA/TMA instructions need it)
- CMake 3.18+
- A CUTLASS **3.x** checkout (header-only; also brings in CuTe from the same repo)
- An NVIDIA **Hopper (sm90)** GPU — the kernel is compiled for `sm90a`
  specifically, which CUTLASS 3.x uses to unlock Hopper-only features

## Get CUTLASS

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v3.5.1   # any 3.x tag
cd ..
```

Place the `cutlass` directory next to this project (or pass
`-DCUTLASS_DIR=/path/to/cutlass` to CMake).

## Build

```bash
mkdir build && cd build
cmake .. -DCUTLASS_DIR=../cutlass
make -j
```

## Run

```bash
./basic_gemm_3x           # uses default M=512 N=256 K=128
./basic_gemm_3x 1024 1024 1024
```

Expected output:

```
M=512 N=256 K=128 : PASSED
```

## How this differs from a CUTLASS 2.0 example

- **Tile/cluster shape instead of a full kernel template.** You specify
  `TileShape` and `ClusterShape` (CuTe `Shape<>` types); `CollectiveBuilder`
  picks the tensor-core instruction, pipeline stage count, and warp-specialized
  schedule for you (`KernelScheduleAuto`, `EpilogueScheduleAuto`).
- **Mainloop and epilogue are separate "collectives"** that get composed into
  a `GemmKernel`, rather than one monolithic device GEMM template.
- **Strides, not just leading dimensions.** `make_cute_packed_stride` builds
  CuTe stride objects from the logical problem shape — the 3.x API is layout-
  and stride-generic via CuTe rather than assuming simple row/column-major.
- **Explicit workspace + two-phase launch.** `can_implement` validates the
  problem against the kernel's constraints, `get_workspace_size` /
  `initialize` / `run` replace the 2.x pattern of just constructing
  `Arguments` and calling the operator.
- **Requires `sm90a`**, not just `sm90` — the "a" suffix is what exposes
  Hopper's warp-specialized/TMA/WGMMA features that this kernel schedule relies on.

## Notes

- This example uses FP16 in/out with FP32 accumulation because that's the
  most common configuration for Hopper tensor-core kernels; swap
  `ElementA`/`ElementB`/`ElementC` for other types (e.g. `cutlass::bfloat16_t`,
  `cutlass::tfloat32_t`) if needed.
- For non-Hopper 3.x targets (e.g. Blackwell `sm100`), the `ArchTag` and
  cluster/tile shapes would need to change accordingly — this example is
  Hopper-specific.

## Reference
The code for this project was developed using Claude
