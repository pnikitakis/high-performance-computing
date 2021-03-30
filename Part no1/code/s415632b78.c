#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define SIZE	4096
#define SHIFTVAL 12 //strength reduction
#define INPUT_FILE	"input.grey"
#define OUTPUT_FILE	"output_sobel.grey"
#define GOLDEN_FILE	"golden.grey"

char horiz_operator[3][3] = {{-1, 0, 1}, 
                             {-2, 0, 2}, 
                             {-1, 0, 1}};
char vert_operator[3][3] = {{1, 2, 1}, 
                            {0, 0, 0}, 
                            {-1, -2, -1}};

double sobel(register unsigned char *input, register unsigned char *output, unsigned char *golden);


const int SIZESQ = SIZE<<SHIFTVAL;//common subexpression + strength reduction
unsigned char input[SIZESQ], output[SIZESQ], golden[SIZESQ];




double sobel(register unsigned char *input, register unsigned char *output, unsigned char *golden)
{
	double PSNR = 0;
	int i, j, k, l, sizeMinus, temp, temp2, temp3, temp4;
	register int res, res2;
	int resX, resY;
	struct timespec  tv1, tv2;
	FILE  * f_in, *f_out, *f_golden;

	sizeMinus = SIZE - 1; 
	memset(output, 0, (sizeof(unsigned char)<<SHIFTVAL));		     //strength reduction
	memset(&output[SIZESQ-SIZE], 0, (sizeof(unsigned char)<<SHIFTVAL)); //strength reduction
	temp = 0;
	for (i = 1; i < sizeMinus; i++) { //common subexpression
		temp = temp + SIZE;
		output[temp] = 0;		//common subexpression + strength reduction
		output[temp + sizeMinus] = 0; //common subexpression
	}

	
	f_in = fopen(INPUT_FILE, "r");
	if (f_in == NULL) {
		printf("File " INPUT_FILE " not found\n");
		exit(1);
	}
  
	f_out = fopen(OUTPUT_FILE, "wb");
	if (f_out == NULL) {
		printf("File " OUTPUT_FILE " could not be created\n");
		fclose(f_in);
		exit(1);
	}  
  
	f_golden = fopen(GOLDEN_FILE, "r");
	if (f_golden == NULL) {
		printf("File " GOLDEN_FILE " not found\n");
		fclose(f_in);
		fclose(f_out);
		exit(1);
	}    

	fread(input, sizeof(unsigned char), SIZESQ, f_in);
	fread(golden, sizeof(unsigned char), SIZESQ, f_golden);
	fclose(f_in);
	fclose(f_golden);
  
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

	temp2 = 0;
	for (i=1; i<sizeMinus; i+=1) {//common subexpression
		temp2 = temp2 + SIZE;  //loop invariant
		for (j=1; j<sizeMinus; j+=1 ) { //common subexpression
			
			
			res = 0;  
			res2 = 0;
			/* Loop fusion */
			temp = temp2 + j; //loop invariant
			
				/* l = -1 */
				temp4 = temp - SIZE;
				temp3 = input[temp4 - 1]; //common subexpression
				    /*k=-1*/
				  res += temp3* horiz_operator[0][0];
				  res2 += temp3* vert_operator[0][0];
				  /*k=0*/
				  temp3 = input[temp4]; //common subexpression
				  res += temp3* horiz_operator[0][1];
				  res2 += temp3* vert_operator[0][1];
				  /*k=1*/
				  temp3 = input[temp4 + 1]; //common subexpression
				  res += temp3* horiz_operator[0][2];
				  res2 += temp3* vert_operator[0][2];																
				
				/* l = 0 */

				temp4 = temp;
				temp3 = input[temp4 - 1]; //common subexpression
				 /*k=-1*/
				res += temp3* horiz_operator[1][0];
				res2 += temp3* vert_operator[1][0];
				/*k=0*/
				temp3 = input[temp4]; //common subexpression
				res += temp3* horiz_operator[1][1];
				res2 += temp3* vert_operator[1][1];
				/*k=1*/
				temp3 = input[temp4 + 1]; //common subexpression
				res += temp3* horiz_operator[1][2];
				res2 += temp3* vert_operator[1][2];
				
			
				/*l = 1 */
			
			
				temp4 = temp + SIZE;
				temp3 = input[temp4 - 1]; //common subexpression
				 /*k=-1*/
				res += temp3* horiz_operator[2][0];
				res2 += temp3* vert_operator[2][0];
				/*k=0*/
				temp3 = input[temp4]; //common subexpression
				res += temp3* horiz_operator[2][1];
				res2 += temp3* vert_operator[2][1];
				/*k=1*/
				temp3 = input[temp4 + 1]; //common subexpression
				res += temp3* horiz_operator[2][2];
				res2 += temp3* vert_operator[2][2];
			
			
			
			
			
			
			
	
			
			
			
			
			
			resX = res*res; 
			
			
			resY = res2*res2;
			
			
			res = (int)sqrt(resX + resY);
			
			if (res > 255)
				output[temp2 + j] = 255;      
			else
				output[temp2 + j] = (unsigned char)res;
		}
	}

	temp2 = 0;					
	for (i=1; i<sizeMinus; i++) {				 //common subexpression
		temp2 = temp2 + SIZE; 				 //loop invariant 
		for ( j=1; j<sizeMinus; j++ ) {			//common subexpression
			temp3 = temp2 + j;     			 //common subexpression
			temp4 = output[temp3] - golden[temp3]; //common subexpression
			PSNR += temp4*temp4; 			//strength reduction				 
		}
	}
  
	PSNR /= (double)(SIZESQ);
	PSNR = 10*log10(65536/PSNR);

	
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("%10g\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

  
	fwrite(output, sizeof(unsigned char), SIZESQ, f_out);
	fclose(f_out);
  
	return PSNR;
}


int main()
{
	double PSNR;
	PSNR = sobel(input, output, golden);
	//printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	//printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");

	return 0;
}

