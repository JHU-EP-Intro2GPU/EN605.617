//EN.605.417.FA
//Bill Burgett
//Sources:  https://github.com/iryont/md5-cracker
//          https://github.com/LuaDist/openssl/blob/master/include/openssl/md5.h

// This is my code that will brute force a password from an md5 hash taken as an input argument at runtime

 #include <stdio.h>
 #include <iostream>
 #include <time.h>
 #include <string.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <sstream>
 
 #include <cuda_runtime.h>
 #include <cuda_runtime_api.h>
 #include <curand_kernel.h>
 #include <device_functions.h>
 
 #define CONST_WORD_LIMIT 10
 #define CONST_CHARSET_LIMIT 100
 
 #define CONST_CHARSET "abcdefghijklmnopqrstuvwxyz"
 #define CONST_CHARSET_LENGTH (sizeof(CONST_CHARSET) - 1)
 
 #define CONST_WORD_LENGTH 8
 
 #define TOTAL_BLOCKS 16384UL
 #define TOTAL_THREADS 512UL
 #define HASHES_PER_KERNEL 128UL
 
 #include "md5.cu"
 
 /* Global variables */
 uint8_t g_wordLength;
 
 char g_word[CONST_WORD_LIMIT];
 char g_charset[CONST_CHARSET_LIMIT];
 char g_cracked[CONST_WORD_LIMIT];
 
 __device__ char g_deviceCharset[CONST_CHARSET_LIMIT];
 __device__ char g_deviceCracked[CONST_WORD_LIMIT];


 __global__ void md5Crack(int wordLength, int offset, uint32_t hash01, uint32_t hash02, uint32_t hash03, uint32_t hash04)
 {
   unsigned int tid = (blockIdx.x * blockDim.x + threadIdx.x)*HASHES_PER_KERNEL;
   unsigned int startTid = tid+(TOTAL_BLOCKS*TOTAL_THREADS*HASHES_PER_KERNEL*offset);
   unsigned int endTid = startTid+HASHES_PER_KERNEL;
   
   /* Thread variables */
   char devPass[CONST_WORD_LIMIT];
   int dWordLength;
   uint32_t dHash01, dHash02, dHash03, dHash04;
   
   /* Copy everything to local memory */
   memcpy(&dWordLength, &wordLength, sizeof(int));
   
    //iterate through each thread's set of hashes to look for match
   for(uint32_t hash = startTid; hash < endTid; hash++){
    //use a combination of modulo and normal division to count up from each thread's unique range and translate that number into a string
     unsigned int quot;
     unsigned int mask;
     unsigned int wordValue=hash;
     for(int i=(dWordLength-1);i>=0;i--){
          mask=1;
          for(int j=0;j<i;j++)
              mask = mask * (CONST_CHARSET_LENGTH);
          quot = wordValue/mask;
          wordValue = wordValue%mask;
          devPass[i]=CONST_CHARSET[quot];
     }
     devPass[dWordLength]='\0';

     md5Hash((unsigned char*)devPass, dWordLength, &dHash01, &dHash02, &dHash03, &dHash04);
 
	 if(dHash01 == hash01 && dHash02 == hash02 && dHash03 == hash03 && dHash04 == hash04){
	   memcpy(g_deviceCracked, devPass, dWordLength);
	 }
   }
 }
 
 int main(int argc, char* argv[]){
   /* Check arguments */
   if(argc != 2 || strlen(argv[1]) != 32){
	 std::cout << argv[0] << " <md5_hash>" << std::endl;
	 return -1;
   }
   
   /* Amount of available devices */
   int devices;
   cudaGetDeviceCount(&devices);
   
   /* Sync type */
   cudaSetDeviceFlags(cudaDeviceScheduleSpin);
   
   /* Display amount of devices */
   std::cout << "Notice: " << devices << " device(s) found" << std::endl;
   
   /* Hash stored as u32 integers */
   uint32_t md5Hash[4];
   
   /* Parse argument */
   for(uint8_t i = 0; i < 4; i++){
	 char tmp[16];
	 
	 strncpy(tmp, argv[1] + i * 8, 8);
	 sscanf(tmp, "%x", &md5Hash[i]);   
	 md5Hash[i] = (md5Hash[i] & 0xFF000000) >> 24 | (md5Hash[i] & 0x00FF0000) >> 8 | (md5Hash[i] & 0x0000FF00) << 8 | (md5Hash[i] & 0x000000FF) << 24;
   }
   
   /* Fill memory */
   memset(g_word, 0, CONST_WORD_LIMIT);
   memset(g_cracked, 0, CONST_WORD_LIMIT);
   memcpy(g_charset, CONST_CHARSET, CONST_CHARSET_LENGTH);
   
   /* Current word length */
   g_wordLength = CONST_WORD_LENGTH;
   
   /* Main device */
   cudaSetDevice(0);
   
   /* Time */
   cudaEvent_t clockBegin;
   cudaEvent_t clockLast;
   
   cudaEventCreate(&clockBegin);
   cudaEventCreate(&clockLast);
   cudaEventRecord(clockBegin, 0);
   
   /* Current word is different on each device */
   char** words = new char*[devices];
   int d_offset;
   
   for(int device = 0; device < devices; device++){
	 cudaSetDevice(device);
	 
	 /* Copy to each device */
	 cudaMemcpyToSymbol(g_deviceCharset, g_charset, sizeof(uint8_t) * CONST_CHARSET_LIMIT, 0, cudaMemcpyHostToDevice);
	 cudaMemcpyToSymbol(g_deviceCracked, g_cracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyHostToDevice);
	 
	 /* Allocate on each device */
     cudaMalloc((void**)&words[device], sizeof(uint8_t) * CONST_WORD_LIMIT);
     cudaMalloc((void**)&d_offset, sizeof(int));
   }
   
   unsigned long offset=0;

   //this will scale the # of loops with respect to the data size.
   unsigned long charSetMaxValue=1;
   for (int i=0; i<CONST_WORD_LENGTH, i++)
        charSetMaxValue*=CONST_CHARSET_LENGTH;

   while((TOTAL_BLOCKS*TOTAL_THREADS*HASHES_PER_KERNEL*offset)<charSetMaxValue)
   {
	    bool found = false;
	 
	    for(int device = 0; device < devices; device++){
	        cudaSetDevice(device);
	   
            /* Copy current data */
            cudaMemcpy(words[device], g_word, sizeof(uint8_t) * CONST_WORD_LIMIT, cudaMemcpyHostToDevice);
            cudaMemcpy(&d_offset, &offset, sizeof(int), cudaMemcpyHostToDevice);
            
            /* Start kernel */
            md5Crack<<<TOTAL_BLOCKS, TOTAL_THREADS>>>(g_wordLength, d_offset, md5Hash[0], md5Hash[1], md5Hash[2], md5Hash[3]);
	    }
	 
        /* Display progress */
        char word[CONST_WORD_LIMIT];

        unsigned int quot;
        unsigned int mask;
        unsigned int wordValue=(TOTAL_BLOCKS*TOTAL_THREADS*HASHES_PER_KERNEL*offset);
        for(int i=(CONST_WORD_LENGTH-1);i>=0;i--){
            mask=1;
            for(int j=0;j<i;j++)
                mask = mask * (CONST_CHARSET_LENGTH);
            quot = wordValue/mask;
            wordValue = wordValue%mask;
            word[i]=CONST_CHARSET[quot];
        }
        word[CONST_WORD_LENGTH]='\0';
            
        std::cout << "Notice: currently at " << std::string(word, g_wordLength) << " (" << (uint32_t)g_wordLength << ")" << std::endl;
        
        for(int device = 0; device < devices; device++)
        {
            cudaSetDevice(device);
        
            /* Synchronize now */
            cudaDeviceSynchronize();
        
            /* Copy result */
            cudaMemcpyFromSymbol(g_cracked, g_deviceCracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyDeviceToHost);//ERROR_CHECK(cudaMemcpyFromSymbol(g_cracked, g_deviceCracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyDeviceToHost)); 
            
            /* Check result */
            if(found = *g_cracked != 0)
            {     
                std::cout << "Notice: cracked " << g_cracked << std::endl; 
                break;
            }
        }
        
        if(found){
            break;
        }
        offset++;
    }
    
    if(!found) std::cout << "Notice: found nothing (host)" << std::endl;
   
   for(int device = 0; device < devices; device++){
	 cudaSetDevice(device);
	 
	 /* Free on each device */
	 cudaFree((void**)words[device]);
   }
   
   /* Free array */
   delete[] words;
   
   /* Main device */
   cudaSetDevice(0);
   
   float milliseconds = 0;
   
   cudaEventRecord(clockLast, 0);
   cudaEventSynchronize(clockLast);
   cudaEventElapsedTime(&milliseconds, clockBegin, clockLast);
   
   std::cout << "Notice: computation time " << milliseconds << " ms" << std::endl;
   
   cudaEventDestroy(clockBegin);
   cudaEventDestroy(clockLast);
 }
 
