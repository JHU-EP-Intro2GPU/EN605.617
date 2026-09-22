#!/usr/bin/env python3
"""
elementwise_add.py

A minimal, real CuTeDSL example: C = A + B on the GPU, written in Python
using NVIDIA's CuTeDSL (the Python embedded DSL shipped with CUTLASS 4.x).

CuTeDSL kernels are ordinary-looking Python functions decorated with
@cute.kernel (device code) and @cute.jit (host code that launches device
code). They are traced and JIT-compiled through the same MLIR -> PTX ->
SASS pipeline as CUTLASS's C++ templates -- this is not an interpreter or
a simulation, it produces and runs a real CUDA kernel.

Install:
    pip install nvidia-cutlass-dsl torch

Run (requires an NVIDIA GPU, CUDA 12.x):
    python3 elementwise_add.py
"""

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# --- Device code -------------------------------------------------------
# @cute.kernel marks this as GPU device code. It reads like a CUDA kernel
# -- thread/block indices via cute.arch, an explicit bounds check -- but
# `a`, `b`, `out` are CuTe Tensors: a pointer plus a Layout, so `a[i]`
# resolves through that layout rather than raw pointer arithmetic.
@cute.kernel
def elem_add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    block_x, _, _ = cute.arch.block_idx()
    block_dim_x, _, _ = cute.arch.block_dim()
    thread_x, _, _ = cute.arch.thread_idx()

    i = block_x * block_dim_x + thread_x
    if i < out.shape[0]:
        out[i] = a[i] + b[i]


# --- Host code -----------------------------------------------------------
# @cute.jit marks this as host code that gets JIT-compiled and is allowed
# to launch @cute.kernel functions via .launch(grid=..., block=...) --
# the same grid/block launch shape as a raw CUDA kernel launch.
@cute.jit
def elem_add(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    n = out.shape[0]
    threads_per_block = 128
    num_blocks = (n + threads_per_block - 1) // threads_per_block

    elem_add_kernel(a, b, out).launch(
        grid=(num_blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def main():
    n = 1_000_000

    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.zeros(n, device="cuda", dtype=torch.float32)

    # from_dlpack wraps existing device tensors (here, PyTorch CUDA
    # tensors) as cute.Tensor objects with no copy -- CuTeDSL kernels can
    # operate directly on tensors from any DLPack-compatible framework.
    a_cute = from_dlpack(a)
    b_cute = from_dlpack(b)
    out_cute = from_dlpack(out)

    # Trace + compile elem_add once for these tensor shapes/dtypes, then
    # invoke it. Subsequent calls with same-shaped tensors reuse the
    # compiled kernel instead of recompiling.
    compiled_elem_add = cute.compile(elem_add, a_cute, b_cute, out_cute)
    compiled_elem_add(a_cute, b_cute, out_cute)

    torch.cuda.synchronize()

    reference = a + b
    max_abs_diff = (out - reference).abs().max().item()

    print(f"Max abs diff vs torch reference: {max_abs_diff}")
    print("PASS" if max_abs_diff < 1e-5 else "FAIL")


if __name__ == "__main__":
    main()
