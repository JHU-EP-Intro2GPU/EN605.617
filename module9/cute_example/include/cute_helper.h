#pragma once
#include "cuda_runtime.h"
#include <cstdlib>
#include <iostream>
#define CUDA_CHECK(status)                                                       \
  {                                                                              \
    cudaError_t error = status;                                                  \
    if (error != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)                   \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;           \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  }

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

inline void device_init(int device_id = 0) {
  CUDA_CHECK(cudaSetDevice(device_id));
}