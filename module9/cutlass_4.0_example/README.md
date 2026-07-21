# CUTLASS 4.0 Example

A minimal standalone project demonstrating a single-precision matrix multiply (SGEMM) with [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) v4.0.0.

The example launches CUTLASS's `cutlass::gemm::device::Gemm` kernel, then verifies the result against a naive reference GEMM on the GPU.

## Requirements

- Linux (or WSL2) with an NVIDIA GPU (compute capability 7.0+)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 11.4 or newer (12.x recommended)
- CMake 3.19+
- A C++17-capable host compiler (GCC 9+, Clang 7+, or MSVC 2019+)

CUTLASS is fetched automatically at configure time via CMake `FetchContent` (tag `v4.0.0`).

> **Note:** This project requires CUDA and an NVIDIA GPU. It cannot be built or run natively on macOS without a remote Linux/CUDA environment.

## Build

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc   # adjust to your CUDA install

mkdir build && cd build
cmake .. -DCUDA_ARCH=80                   # 80=Ampere, 89=Ada, 90=Hopper
cmake --build . -j
```

Set `CUDA_ARCH` to match your GPU:

| GPU | `CUDA_ARCH` |
|-----|-------------|
| V100 | 70 |
| A100 | 80 |
| RTX 4090 | 89 |
| H100 | 90 |

## Run

```bash
./basic_gemm              # 128x128x128 GEMM, alpha=1, beta=0
./basic_gemm 512 512 512  # custom M N K
./basic_gemm 256 256 256 2.0 0.5  # M N K alpha beta
```

Expected output on success:

```
Passed.
```

## Project layout

```
cutlass-4-example/
├── CMakeLists.txt       # Fetches CUTLASS 4.0 and builds basic_gemm
├── include/helper.h     # CUDA/CUTLASS error-check macros
└── src/basic_gemm.cu    # SGEMM example + reference verifier
```

## Next steps

- Browse the [CUTLASS 4.0 examples](https://github.com/NVIDIA/cutlass/tree/v4.0.0/examples) for tensor cores, mixed precision, and convolutions.
- See the [CUTLASS documentation](https://docs.nvidia.com/cutlass/) for CuTe abstractions introduced in 3.x/4.x.


## Reference
The code for this project was developed using Cursor AI