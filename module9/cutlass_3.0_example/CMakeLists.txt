cmake_minimum_required(VERSION 3.18)
project(cutlass3_basic_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Point this at a checkout of https://github.com/NVIDIA/cutlass
# (use a 3.x tag/branch, e.g. `git checkout v3.5.1`). CUTLASS 3.0 is
# header-only, but it also pulls in CuTe, which ships in the same repo.
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/cutlass" CACHE PATH "Path to CUTLASS repository")

if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
        "CUTLASS headers not found at ${CUTLASS_DIR}/include. "
        "Clone https://github.com/NVIDIA/cutlass next to this project "
        "(or pass -DCUTLASS_DIR=/path/to/cutlass) and check out a 3.x release.")
endif()

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "CUTLASS 3.x Hopper kernels require CUDA 12.0 or newer (found ${CUDAToolkit_VERSION}).")
endif()

add_executable(basic_gemm_3x basic_gemm_3x.cu)

target_include_directories(basic_gemm_3x PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

target_link_libraries(basic_gemm_3x PRIVATE CUDA::cudart)

# CUTLASS 3.0's warp-specialized Hopper kernels need sm90a specifically
# (the "a" suffix enables architecture-specific features like WGMMA/TMA
# that plain sm_90 does not expose).
set_property(TARGET basic_gemm_3x PROPERTY CUDA_ARCHITECTURES 90a)

target_compile_options(basic_gemm_3x PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -extended-lambda>
)
