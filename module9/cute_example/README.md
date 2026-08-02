# CuTe Example Project
A minimal standalone project demonstrating [CuTe](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html) — NVIDIA's C++ tensor algebra library shipped with [CUTLASS](https://github.com/NVIDIA/cutlass) v4.0.
CuTe provides `Layout` and `Tensor` abstractions for describing multidimensional data and thread hierarchies. These examples show the core concepts before diving into full GEMM kernels.
## Examples
| Target | Description |
|--------|-------------|
| `layout_demo` | Creates row/column-major layouts, prints coordinate-to-index mapping, runs a simple GPU vector add using CuTe tensors |
| `cute_sgemm` | Implements `C = A * B^T` using CuTe tiling, `copy`, and `gemm` (adapted from the official `sgemm_1.cu` tutorial) |
## Requirements
- Linux (or WSL2) with an NVIDIA GPU (compute capability 7.0+)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 11.4+ (12.x recommended)
- CMake 3.19+
- C++17 host compiler
CUTLASS v4.0.0 is fetched automatically at configure time.
> **Note:** Requires CUDA and an NVIDIA GPU. Cannot build or run natively on macOS.
## Build
```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
mkdir build && cd build
cmake .. -DCUDA_ARCH=80
cmake --build . -j
```
## Run
```bash
./layout_demo
./cute_sgemm           # 256x256x256 by default
./cute_sgemm 512 512 512
```
## Project layout
```
├── CMakeLists.txt
├── include/cute_helper.h
└── src/
    ├── layout_demo.cu   # Layout/Tensor basics
    └── cute_sgemm.cu    # Tiled SGEMM with CuTe
```
## Learn more
- [Getting Started With CuTe](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html)
- [CuTe GEMM Tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)
- [CUTLASS CuTe examples](https://github.com/NVIDIA/cutlass/tree/v4.0.0/examples/cute)

## Reference
The code for this project was developed using Cursor AI