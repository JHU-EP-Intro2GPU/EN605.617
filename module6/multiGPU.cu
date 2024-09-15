/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

// This application demonstrates how to use the CUDA api to use multiple GPUs.
//
// There's one CUDA context per thread. To use multiple CUDA contexts you
// have to create multiple threads. One for each GPU. For optimal performance,
// the number of CPU cores should be equal to the number of GPUs in the system.
//
// Creating CPU threads has a certain overhead. So, this is only worth when you
// have a significant amount of work to do per thread. It's also recommended to
// create a pool of threads and reuse them to avoid this overhead.
//
// Note that in order to detect multiple GPUs in your system you have to disable
// SLI in the nvidia control panel. Otherwise only one GPU is visible to the 
// application. On the other side, you can still extend your desktop to screens 
// attached to both GPUs.


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <helper_cuda.h>
#include <multithreading.h>

// includes, project

// Maximum number of CPU threads or GPUs.
#define MAX_CPU_THREAD    4

// Kernel configuration.
// use small number of blocks for device emulation to ensure we don't run too long.
#ifdef __DEVICE_EMULATION__ 
#define NUM_THREADS    256
#define NUM_BLOCKS     1
#else
#define NUM_THREADS    256
#define NUM_BLOCKS    1024*32
#endif

static int s_gpuCount = 0;

////////////////////////////////////////////////////////////////////////////////
// Dummy kernel
////////////////////////////////////////////////////////////////////////////////
__global__ static void kernel(float * g_idata, float * g_odata)
{
    extern  __shared__  float sdata[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int dim = blockDim.x;

    sdata[tid] = g_idata[tid + dim * bid];

    // @@ perform some useful computations here!

    for (int i = 0; i < 4 * dim; i++)
    {
        __syncthreads();
        
        sdata[tid] = sdata[tid] + 2.0f * sdata[(tid ^ i) & (dim - 1)];
    }

    __syncthreads();

    g_idata[tid + dim * bid] = sdata[tid];
}


////////////////////////////////////////////////////////////////////////////////
// GPU thread
////////////////////////////////////////////////////////////////////////////////
static CUT_THREADPROC gpuThread(int * device)
{
    cudaSetDevice(*device);

    const int mem_size = NUM_BLOCKS * NUM_THREADS * sizeof(float) / s_gpuCount;

    float * idata;
    cudaMalloc( (void**) &idata, mem_size);

    float * odata;
    cudaMalloc( (void**) &odata, mem_size);

    // @@ Copy some values to the buffers.

    // Invoke kernel on this device.
    kernel<<<NUM_BLOCKS / s_gpuCount, NUM_THREADS, NUM_THREADS*sizeof(float)>>>(idata, odata);

    // @@ Get the results back.

    CUT_THREADEND;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{  

    // Enumerate GPUs.
    cudaGetDeviceCount(&s_gpuCount);

    unsigned int timer = 0;

    // Cap the number of threads.
    if (s_gpuCount > MAX_CPU_THREAD)
    {
        s_gpuCount = MAX_CPU_THREAD;
    }

    if (s_gpuCount == 0)
    {
        printf("No GPU found\n");
    }
    else if (s_gpuCount == 1)
    {
        printf("Only one GPU found\n");


        // Run a single thread.
        int thread = 0;
        gpuThread(&thread);

    }
//    else
//    {
//       int threadIds[MAX_CPU_THREAD];
//
//        printf("%d GPUs found\n", s_gpuCount);
    

//        CUTThread * threads = (CUTThread *)malloc(sizeof(CUTThread) * s_gpuCount);

        // Start one thread for each device.
//        for(int i = 0; i < s_gpuCount; i++)
//        {
//           threadIds[i] = i;
//            threads[i] = cutStartThread((CUT_THREADROUTINE)gpuThread, (void *)&threadIds[i]);
//        }

        // Wait for all the threads to finish.
//        cutWaitForThreads(threads, s_gpuCount);

//        free(threads);

//    }

    printf("Test PASSED\n");

}
