#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_functions.h"
#include "helper_cuda.h"

#include <ctime>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cufft.h>
#include <fstream>

using namespace std;
typedef float2 Complex;


//Found at http://techqa.info/programming/question/36889333/cuda-cufft-2d-example


__global__ void ComplexMUL(Complex *a, Complex *b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    a[i].x = a[i].x * b[i].x - a[i].y*b[i].y;
    a[i].y = a[i].x * b[i].y + a[i].y*b[i].x;
}


int main()
{


    int N = 5;
    int SIZE = N*N;


    Complex *fg = new Complex[SIZE];
    for (int i = 0; i < SIZE; i++){
        fg[i].x = 1;
        fg[i].y = 0;
    }
    Complex *fig = new Complex[SIZE];
    for (int i = 0; i < SIZE; i++){
        fig[i].x = 1; // 
        fig[i].y = 0;
    }
    for (int i = 0; i < N * N; i = i + N)
    {
        for (int j=0; j < N; j++){
            cout << fg[i+j].x << " ";
        }
        cout << endl;
    }
    cout << "----------------" << endl;
    for (int i = 0; i < N * N; i = i + N)
    {
        for (int j=0; j < N; j++){
            cout << fig[i+j].x << " ";
        }
        cout << endl;
    }
    cout << "----------------" << endl;

    int mem_size = sizeof(Complex)* SIZE;


    cufftComplex *d_signal;
    checkCudaErrors(cudaMalloc((void **) &d_signal, mem_size)); 
    checkCudaErrors(cudaMemcpy(d_signal, fg, mem_size, cudaMemcpyHostToDevice));

    cufftComplex *d_filter_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_filter_kernel, mem_size));
    checkCudaErrors(cudaMemcpy(d_filter_kernel, fig, mem_size, cudaMemcpyHostToDevice));

    // cout << d_signal[1].x << endl;
    // CUFFT plan
    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_C2C);

    // Transform signal and filter
    printf("Transforming signal cufftExecR2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex *)d_filter_kernel, (cufftComplex *)d_filter_kernel, CUFFT_FORWARD);

    printf("Launching Complex multiplication<<< >>>\n");
    ComplexMUL <<< N, N >> >(d_signal, d_filter_kernel);

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);

    Complex *result = new Complex[SIZE];
    cudaMemcpy(result, d_signal, sizeof(Complex)*SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i = i + N)
    {
        for (int j=0; j < N; j++){
            cout << result[i+j].x << " ";
        }
        cout << endl;
    }

    delete result, fg, fig;
    cufftDestroy(plan);
    //cufftDestroy(plan2);
    cudaFree(d_signal);
    cudaFree(d_filter_kernel);

}