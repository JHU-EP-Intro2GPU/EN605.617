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

__constant__  static char hashKey[HASH_SIZE];
__constant__  static char plainTextPwd[PWD_SIZE];

__device__ unsigned char* doTheHash(unsigned char* inputString){
	unsigned char* hashVal;
	uint32_t* a1;
	uint32_t* b1;
	uint32_t* c1;
	uint32_t* d1;
	uint32_t temp[4];
	md5Hash(inputString,(uint32_t)32,a1,b1,c1,d1);
	printf("%s,%s,%s,%s\n",a1,b1,c1,d1);
	memcpy(&temp[0],&a1,sizeof(uint32_t));
	memcpy(&temp[1],&b1,sizeof(uint32_t));
	memcpy(&temp[2],&c1,sizeof(uint32_t));
	memcpy(&temp[3],&d1,sizeof(uint32_t));
	Encode(hashVal,temp,4);
	return hashVal;	
}

__device__ int my_strcmp(const char *str_a, const char *str_b, unsigned len = 256){
  int match = 0;
  unsigned i = 0;
  unsigned done = 0;
  while ((i < len) && (match == 0) && !done){
    if ((str_a[i] == 0) || (str_b[i] == 0)) done = 1;
    else if (str_a[i] != str_b[i]){
      match = i+1;
      if (((int)str_a[i] - (int)str_b[i]) < 0)
		  match = 0-(i+1);
	}
    i++;}
  return match;
  }

/*
* This is the function that will actually perform the hashing on the GPU. It will use the thread id to
* determine the string to test. For proof-of-concept I have limited the character set to just integers for now.
* The thread id will correspond to a string built out of an input character set. For example, if the character
* set was 26 letters of the alphabet and the tid was 20000, then the resultant string would be "acoe"
*/
__global__ void deviceHash(const char* charSet, int charSetSize, unsigned char devPass[maxPwdLen+1], int offset, int* foundFlag)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	//MD5 md5;
	//char hashKey[32];
	//char* tempHash = md5.digestString("00043110");
	//memcpy(hashKey,tempHash,32);
	//char* charSet = "0123456789";//"abcdefghijklmnopqrstuvwxyzABCEDFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()";
	//char tryPass[maxPwdLen+1];// = (char*)malloc(sizeof(char)*maxPwdLen);
	unsigned int quot;
	unsigned int mask;
	unsigned char* tempHash;
	//int charSetSize=strlen(charSet);
	unsigned int wordValue=tid;
	
	for(int i=0;i<maxPwdLen;i++){
		mask=1;
		for (int j=0;j<(maxPwdLen-i-1);j++)
			mask*=charSetSize;
		quot = wordValue/mask;
		wordValue = wordValue%mask;
		//tryPass[i]=charSet[quot];
		devPass[i]=charSet[quot];
	}
	devPass[maxPwdLen]='\0';
	//tryPass[maxPwdLen]='\0';
	
	tempHash = doTheHash(devPass);//md5.digestString( devPass );//tryPass );
	
	//if(my_strcmp((char*)devPass,hashKey)==0)
	//	foundFlag[0]=1;
		
	__syncthreads();
}

__global__ void testHash(char* testPrint)
{
	
	unsigned char* tempHash;
	tempHash = (char*)doTheHash((unsigned char*)"aaaaaacd");
	//cuPrintf("test hash %s\n",tempHash);
		
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
	const unsigned int num_threads = 128;
	const unsigned int num_blocks = WORK_SIZE / num_threads;
	float delta;
	const char* charSet = "abcdefghijklmnopqrstuvwxyz";//ABCEDFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()";
	unsigned char crackedPass[maxPwdLen+1];
	unsigned char* devPass;
	
	int* foundFlag;
	cudaMallocManaged(&foundFlag, sizeof(int));
	foundFlag[0]=-1;
	
	char* testPrint;
	cudaMallocManaged(&testPrint, sizeof(char)*32);
	testPrint[0]='n';
	
	//load hash to crack into constant mem 
	MD5 md5;
	char* tempHash = md5.digestString(inputPlainText);
	cudaMemcpyToSymbol(hashKey, &tempHash, (sizeof(char)*HASH_SIZE));
	printf("Input text is: %s\nCPU hash value is: %s -%s\n",inputPlainText,hashKey,tempHash);
 	
 	cudaMalloc(&devPass, (sizeof(char)*HASH_SIZE));
	
	cudaEvent_t start_time = get_time();
	//int i=0;
	//while ( (i<2) && (foundFlag[0]!=1) ){
		//deviceHash << <num_blocks, num_threads >> >(charSet,strlen(charSet),devPass,i,foundFlag);
		deviceHash << <1, 1>> >(charSet,26,devPass,0,foundFlag);
		cudaThreadSynchronize();	// Wait for the GPU launched work to complete
		//i++;
	//}	
	cudaGetLastError();
	
	testHash << <1, 1>> >(testPrint);
	cudaThreadSynchronize();
	cudaGetLastError();
	printf("testPrint is: %s\n",testPrint);
	
	//stop the clock
	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	
	//if(foundFlag[0]==1){
		cudaMemcpy(crackedPass, devPass, sizeof(char) * maxPwdLen+1, cudaMemcpyDeviceToHost);
		printf("Password Cracked! Plain-text is: %s\n", crackedPass);
	//} else {
	//	printf("No dice, couldn't crack the pass withing current parameters\n");
	//}

	//find execution time
	cudaEventElapsedTime(&delta, start_time, end_time);

	cudaFree(crackedPass);
	cudaFree(foundFlag);
	cudaFree(testPrint);
	cudaDeviceReset();

	return delta;
}

int main()
{
	printf("in main\n");
	float elapsedTime = runHash("aaaaaacd");//password");
	printf("Execution time = %f\n",elapsedTime);	

	return 0;
}
