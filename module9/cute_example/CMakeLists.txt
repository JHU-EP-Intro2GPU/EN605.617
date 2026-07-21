cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(cute_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

set(CUTLASS_INCLUDE_DIRS
  "${cutlass_SOURCE_DIR}/include"
  "${cutlass_SOURCE_DIR}/tools/util/include"
)

function(add_cute_example target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE
    ${CUTLASS_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  target_compile_options(${target} PRIVATE
    --expt-relaxed-constexpr
    --extended-lambda
  )
  set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${target} PRIVATE CUDA::cudart)
endfunction()

add_cute_example(layout_demo src/layout_demo.cu)
add_cute_example(cute_sgemm src/cute_sgemm.cu)