// EN.605.417
// Bill Burgett

#include <stdio.h>
#include "md5.cu"
#include "md5.h"

#define WORK_SIZE 128
#define	HASH_SIZE 32 //size of an md5 hash is 32 characters
#define PWD_SIZE 8
#define maxPwdLen 8

using namespace std;


__global__ void doTheHash(){//uint32_t* a1, uint32_t* b1, uint32_t* c1, uint32_t* d1){
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	/*unsigned char* hashVal;
	uint32_t* a1;
	uint32_t* b1;
	uint32_t* c1;
	uint32_t* d1;*/
	//char temp[9] = "aaaaaacd";
	//uint32_t wordlength=8;
	//uint32_t th1,th2,th3,th4;
	//md5Hash((unsigned char*)temp,wordlength,&th1,&th2,&th3,&th4);
	/*memcpy(&a1,&th1,sizeof(uint32_t));
	memcpy(&b1,&th2,sizeof(uint32_t));
	memcpy(&c1,&th3,sizeof(uint32_t));
	memcpy(&d1,&th4,sizeof(uint32_t));*/
	/*uint32_t temp[4];
	memcpy(&temp[0],&a1,sizeof(uint32_t));
	memcpy(&temp[1],&b1,sizeof(uint32_t));
	memcpy(&temp[2],&c1,sizeof(uint32_t));
	memcpy(&temp[3],&d1,sizeof(uint32_t));
	Encode(hashVal,temp,4);
	return hashVal;	*/
}

/*__global__ void testHash(char* testPrint)
{
	
	//unsigned char* tempHash;
	char* temp = "aaaaaacd";
	testPrint = (char*)doTheHash((unsigned char*)temp);
	//cuPrintf("test hash %s\n",tempHash);
		
	__syncthreads();
}*/


//First cut at the wrapper to run the kernel
__host__ void runHash(char* inputPlainText)
{
	/*char* testPrint;
	cudaMalloc(&testPrint, sizeof(char)*32);
	char* outp;*/

	uint32_t* a1;
	uint32_t* b1;
	uint32_t* c1;
	uint32_t* d1;
	cudaMalloc(&a1, sizeof(uint32_t) );
	cudaMalloc(&b1, sizeof(uint32_t) );
	cudaMalloc(&c1, sizeof(uint32_t) );
	cudaMalloc(&d1, sizeof(uint32_t) );
	
	uint32_t* a2;
	uint32_t* b2;
	uint32_t* c2;
	uint32_t* d2;
	int numblocks=1;
	int numthreads=1;
	
	//load hash to crack into constant mem 
	printf("initialized vars\n");
	MD5 md5;
	char* tempHash;
	tempHash = md5.digestString(inputPlainText);
	printf("Input text is: %s\nCPU hash value is: %s\n",inputPlainText,tempHash);
 		
	doTheHash << <numblocks, numthreads >> >();//a1,b1,c1,d1);
	cudaThreadSynchronize();
	cudaGetLastError();
	//cudaMemcpy(outp, testPrint, (sizeof(char)*HASH_SIZE), cudaMemcpyDeviceToHost);
	cudaMemcpy(&a2, &a1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(&b2, &b1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(&c2, &c1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(&d2, &d1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	
	//printf("testPrint is: %s\n",outp);
	printf("testPrint is: %i, %i, %i, %i\n",a2[0],b2[0],c2[0],d2[0]);
	
	//cudaFree(testPrint);
	cudaFree(a1);
	cudaFree(b1);
	cudaFree(c1);
	cudaFree(d1);
	cudaFree(a2);
	cudaFree(b2);
	cudaFree(c2);
	cudaFree(d2);
	cudaDeviceReset();
}

int main()
{
	printf("in main\n");
	runHash("aaaaaacd");//password");

	return 0;
}
