cmake_minimum_required(VERSION 3.18)
project(cutlass_basic_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Point this at a checkout of https://github.com/NVIDIA/cutlass
# (use a 2.x tag/branch, e.g. `git checkout v2.11.0`). CUTLASS 2.0 is
# header-only, so nothing needs to be built or linked from the repo itself.
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/cutlass" CACHE PATH "Path to CUTLASS repository")

if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
        "CUTLASS headers not found at ${CUTLASS_DIR}/include. "
        "Clone https://github.com/NVIDIA/cutlass next to this project "
        "(or pass -DCUTLASS_DIR=/path/to/cutlass) and check out a 2.x release.")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(basic_gemm basic_gemm.cu)

target_include_directories(basic_gemm PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

target_link_libraries(basic_gemm PRIVATE CUDA::cudart)

# Adjust to match your GPU(s): 70=Volta, 75=Turing, 80/86=Ampere, 89=Ada, 90=Hopper
set_property(TARGET basic_gemm PROPERTY CUDA_ARCHITECTURES 70 75 80)

target_compile_options(basic_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
