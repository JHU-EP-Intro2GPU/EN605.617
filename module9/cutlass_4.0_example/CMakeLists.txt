cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(cutlass_4_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Override with -DCUDA_ARCH=89 for Ada, 90 for Hopper, etc.
if(NOT DEFINED CUDA_ARCH)
  set(CUDA_ARCH 80 CACHE STRING "Target CUDA compute capability (e.g. 80, 89, 90)")
endif()

include(FetchContent)
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG v4.0.0
  GIT_SHALLOW TRUE
)
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  FetchContent_Populate(cutlass)
endif()

add_executable(basic_gemm src/basic_gemm.cu)
target_include_directories(basic_gemm PRIVATE
  "${cutlass_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(basic_gemm PRIVATE
  --expt-relaxed-constexpr
  --extended-lambda
)
set_property(TARGET basic_gemm PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(basic_gemm PRIVATE CUDA::cudart)
