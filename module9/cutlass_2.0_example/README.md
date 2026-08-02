# CUTLASS 2.0 Basic GEMM Example

A minimal project showing how to call a CUTLASS 2.0 device-level GEMM
(`cutlass::gemm::device::Gemm`) to compute `C = alpha * A * B + beta * C`
in single precision, with a naive CPU reference check.

## Requirements

- CUDA Toolkit (nvcc), 10.2+ recommended
- CMake 3.18+
- A CUTLASS 2.x checkout (header-only, no build/install step needed)
- An NVIDIA GPU with compute capability matching the architectures set
  in `CMakeLists.txt` (defaults to 70/75/80 — Volta/Turing/Ampere)

## Get CUTLASS

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v2.11.0   # any 2.x tag
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
./basic_gemm           # uses default M=512 N=256 K=128
./basic_gemm 1024 1024 1024
```

Expected output:

```
M=512 N=256 K=128 : PASSED
```

## Notes

- CUTLASS 2.0 is header-only, so the "library" is just the `include/`
  directory — there's nothing to compile/link beyond your own `.cu` file.
- `cutlass::gemm::device::Gemm<...>` is templated on element type and
  layout for A, B, and C; this example uses `float` + `ColumnMajor`
  for all three, which is the simplest configuration.
- To target a specific GPU, edit `CUDA_ARCHITECTURES` in
  `CMakeLists.txt` (e.g. `80` for Ampere, `90` for Hopper).
- For higher performance you'd typically also specify a `ThreadblockShape`,
  `WarpShape`, and `InstructionShape` template arguments, or use tensor
  cores via `cutlass::arch::OpClassTensorOp` — this example intentionally
  sticks to the defaults for clarity.

## Reference
The code for this project was developed using Claude