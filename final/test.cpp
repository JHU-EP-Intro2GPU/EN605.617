#include "md5.h"
#include <String.h>
#include <stdio.h>
#include <math.h>

int main()
{
	MD5 md5 ;
	puts( md5.digestString( "password" ) ) ;

	for(int i=(8-1);i>=0;i--)
		printf("%i\n",i);
	
	char* charSet = "abcdefghijklmnopqrstuvwxyzABCEDFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()";
	char* tryPass[8];
	int tid=126;
	for(int i=0;i<8;i++){
		printf("i=%i,%i   -   ",(8-i),(tid%((int)pow(72, (double)(8-i-1) ))));
		int mask = ((int)pow(72, (double)(8-i) ));
		printf("mask=%i\n",mask);
		tryPass[8-i-1]=&charSet[(tid/mask)];
	}
	printf("%s\n",tryPass);
	//puts(md5.digestString(*tryPass));
	
	return 0;
}
