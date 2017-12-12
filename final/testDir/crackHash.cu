// EN.605.417
// Bill Burgett

#include <stdio.h>
#include "md5.h"

#define WORK_SIZE 128
#define	HASH_SIZE 32 //size of an md5 hash is 32 characters
#define PWD_SIZE 8
#define maxPwdLen 8

using namespace std;

__constant__  static char hashKey[HASH_SIZE];
__constant__  static char plainTextPwd[PWD_SIZE];

/*
* This is the function that will actually perform the hashing on the GPU. It will use the thread id to
* determine the string to test. For proof-of-concept I have limited the character set to just integers for now.
* The thread id will correspond to a string built out of an input character set. For example, if the character
* set was 26 letters of the alphabet and the tid was 20000, then the resultant string would be "acoe"
*/
__global__ void deviceHash(const char* charSet, int charSetSize, char devPass[maxPwdLen+1], int offset, int* foundFlag)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	MD5 md5;
	//char hashKey[32];
	//char* tempHash = md5.digestString("00043110");
	//memcpy(hashKey,tempHash,32);
	//char* charSet = "0123456789";//"abcdefghijklmnopqrstuvwxyzABCEDFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()";
	//char tryPass[maxPwdLen+1];// = (char*)malloc(sizeof(char)*maxPwdLen);
	unsigned int quot;
	unsigned int mask;
	char* tempHash;
	//int charSetSize=strlen(charSet);
	unsigned int wordValue=tid;
	
	for(int i=0;i<maxPwdLen;i++){
		mask = ((unsigned int)(pow(charSetSize,(maxPwdLen-i-1))));
		quot = wordValue/mask;
		wordValue = wordValue%mask;
		//tryPass[i]=charSet[quot];
		devPass[i]=charSet[quot];
	}
	devPass[maxPwdLen]='\0';
	//tryPass[maxPwdLen]='\0';
	
	tempHash = md5.digestString( devPass );//tryPass );
	
	if(strcmp(devPass,hashKey)==0)
		foundFlag[0]=1;
		
	__syncthreads();
}

__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

//First cut at the wrapper to run the kernel
__host__ float runHash(char* inputPlainText)
{
	MD5 md5;
	const unsigned int num_threads = 128;
	const unsigned int num_blocks = WORK_SIZE / num_threads;
	float delta;
	const char* charSet = "abcdefghijklmnopqrstuvwxyz";//ABCEDFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()";
	char crackedPass[maxPwdLen+1];
	char* devPass;
	int* foundFlag;
	
	cudaMallocManaged(&foundFlag, sizeof(int));
	foundFlag[0]=-1;
	
	//load hash to crack into constant mem 
	char* tempHash = md5.digestString(inputPlainText);
	cudaMemcpyToSymbol(hashKey, &tempHash, (sizeof(char)*HASH_SIZE));
 	
 	cudaMalloc(&devPass, (sizeof(char)*HASH_SIZE));
	
	cudaEvent_t start_time = get_time();

	int i=0;
	while ( (i<2) && (foundFlag[0]!=1) ){
		deviceHash << <num_blocks, num_threads >> >(charSet,strlen(charSet),devPass,i,foundFlag);
		cudaThreadSynchronize();	// Wait for the GPU launched work to complete
		i++;
	}	
	cudaGetLastError();

	//stop the clock
	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	
	if(foundFlag[0]==1){
		cudaMemcpy(crackedPass, devPass, sizeof(char) * maxPwdLen+1, cudaMemcpyDeviceToHost);
		printf("Password Cracked! Plain-text is: %s\n", crackedPass);
	} else {
		printf("No dice, couldn't crack the pass withing current parameters\n");
	}

	//find execution time
	cudaEventElapsedTime(&delta, start_time, end_time);

	cudaFree(crackedPass);
	cudaFree(foundFlag);
	cudaDeviceReset();

	return delta;
}

int main()
{
	float elapsedTime = runHash("aaaaaacd");//password");
	printf("Execution time = %f\n",elapsedTime);	

	return 0;
}
