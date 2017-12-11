#include "md5.h"
#include <string>
#include <stdio.h>
#include <math.h>
using namespace std;
//c++ tester that I use to try out stuff I want to add. Not part of submission.
/*
1 - I'm starting from known input length & data set
2 - Find # of potential combos from length & data set
3 - find # of segments needed by div combos by uintmax
4 - iterate through kernel using 1 seg at a time to find hash string
	use shared mem to store result flag
*/
//since pwd length is known, a tid of 0 will correspond to string = "00000000"

int main()
{
	MD5 md5;
	char hashKey[32];
	char* tempHash = md5.digestString("00043110");
	memcpy(hashKey,tempHash,32);
	printf("%s - %s\n",tempHash,hashKey);
	
	int maxPwdLen=8;
	char* charSet = "0123456789";//"abcdefghijklmnopqrstuvwxyzABCEDFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()";
	char tryPass[maxPwdLen+1];// = (char*)malloc(sizeof(char)*maxPwdLen);
	char* answer="dunno";
	//unsigned int tid=126;
	unsigned int quot;
	unsigned int mask;
	//char* tempHash;
	unsigned int wordValue;
	string holder;
	//unsigned int tid=12345;
	for(unsigned int tid=0;tid<100000000;tid++)
	{
		wordValue=tid;
		for(int i=0;i<maxPwdLen;i++){
			mask = ((unsigned int)(pow(10,(maxPwdLen-i-1))));
			quot = wordValue/mask;
			wordValue = wordValue%mask;
			//printf("rem=%u, quot=%u, mask=%u, tid=%u\n",tid,quot,mask,tid);
			tryPass[i]=charSet[quot];
		}
		tryPass[maxPwdLen]='\0';
		tempHash = md5.digestString( tryPass );
		//printf("%s\n%s - %s\n",tryPass,tempHash,hashKey);
		if(strcmp(tempHash,hashKey)==0){
			answer=tryPass;
			break;
		}
	}
	
	printf("%s\n",answer);

	
	//puts(md5.digestString(*tryPass));
	
	return 0;
}
