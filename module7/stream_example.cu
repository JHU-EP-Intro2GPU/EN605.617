/*https://cdac.in/index.aspx?id=ev_hpc_gpu-comp-nvidia-cuda-streams#hetr-cuda-prog-cuda-streams*/

#include <stdio.h> 
#include <time.h> 
#include <cuda.h> 

#define sizeOfArray 1024*1024
 
__global__ void arrayAddition(int *device_a, int *device_b, int *device_result)
{

	int threadId = threadIdx.x + blockIdx.x * blockDim.x ;

	if (threadId < sizeOfArray) 
        device_result[threadId]= device_a[threadId]+device_b[threadId]; 
} 


/* Check for safe return of all calls to the device */ 

int main ( int argc, char **argv ) 
{ 

  cudaDeviceProp prop; 
  int *host_a, *host_b, *host_result; 
  int *device_a, *device_b, *device_result; 
  int whichDevice; 

  cudaGetDeviceCount( &whichDevice); 
  cudaGetDeviceProperties( &prop, whichDevice); 

  cudaEvent_t start, stop; 
  float elapsedTime; 

  cudaEventCreate( &start ); 
  cudaEventCreate( &stop ); 

  cudaStream_t stream; 
  cudaStreamCreate(&stream); 

  cudaMalloc( ( void**)& device_a, sizeOfArray * sizeof ( *device_a ) ); 
  cudaMalloc( ( void**)& device_b,sizeOfArray * sizeof ( *device_b ) ); 
  cudaMalloc( ( void**)& device_result, sizeOfArray * sizeof ( *device_result ) ); 

  cudaHostAlloc((void **)&host_a, sizeOfArray*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&host_b, sizeOfArray*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&host_result, sizeOfArray*sizeof(int), cudaHostAllocDefault);

  for(int index = 0; index < sizeOfArray; index++) 
  { 
   host_a[index] = rand()%10; 
   host_b[index] = rand()%10; 
  } 

  cudaEventRecord(start);

  cudaMemcpyAsync(device_a, host_a,sizeOfArray * sizeof ( int ), cudaMemcpyHostToDevice, stream); 

  cudaMemcpyAsync(device_b, host_b, sizeOfArray * sizeof ( int ), cudaMemcpyHostToDevice, stream); 

  /*Kernel call*/ 

  arrayAddition <<<sizeOfArray, 1, 1, stream>>>(device_a, device_b, device_result);

  cudaMemcpyAsync(host_result, device_result, sizeOfArray * sizeof ( int ), cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  cudaEventElapsedTime(&elapsedTime, start, stop); 

  printf("*********** CDAC - Tech Workshop : hyPACK-2013 \n"); 
  printf("\n Size of array : %d \n", sizeOfArray); 
  printf("\n Time taken: %3.1f ms \n", elapsedTime); 

  cudaFreeHost(host_a); 
  cudaFreeHost(host_b); 
  cudaFreeHost(host_result); 
  cudaFree(device_a); 
  cudaFree(device_b); 
  cudaFree(device_result); 

  return 0; 
}
