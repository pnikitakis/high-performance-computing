#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define SIZE	4096
#define INPUT_FILE	"input.grey"
#define OUTPUT_FILE	"output_sobel.grey"
#define GOLDEN_FILE	"golden.grey"

char horiz_operator[3][3] = {{-1, 0, 1}, 
                             {-2, 0, 2}, 
                             {-1, 0, 1}};
char vert_operator[3][3] = {{1, 2, 1}, 
                            {0, 0, 0}, 
                            {-1, -2, -1}};

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden);


const int SIZESQ = SIZE*SIZE;//common subexpression
unsigned char input[SIZESQ], output[SIZESQ], golden[SIZESQ];




double sobel(unsigned char *input, unsigned char *output, unsigned char *golden)
{
	double PSNR = 0, t;
	int i, j, k, l, sizeMinus, temp, temp2, temp3;
	unsigned int p;
	int res, res2, resX, resY;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;

	sizeMinus = SIZE - 1; 
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < sizeMinus; i++) { //common subexpression
		output[i*SIZE] = 0;
		output[i*SIZE + sizeMinus] = 0; //common subexpression
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

	for (i=1; i<sizeMinus; i+=1) {//common subexpression
		temp2 = i*SIZE;  //loop invariant
		for (j=1; j<sizeMinus; j+=1 ) { //common subexpression
			
			
			res = 0;  
			res2 = 0;
			/* Loop fusion */
			temp = temp2 + j; //loop invariant
			for (l = -1; l <= 1; l++) {  /* interchange loop */
				for (k = -1; k <= 1; k++) {
					temp3 = input[temp + l*SIZE + k]; //common subexpression
					res += temp3* horiz_operator[l+1][k+1];
					res2 += temp3* vert_operator[l+1][k+1];
				}
			}					
			
			resX = pow(res,2); 
			
			
			resY = pow(res2,2);
			
			
			res = (int)sqrt(resX + resY);
			
			if (res > 255)
				output[temp2 + j] = 255;      
			else
				output[temp2 + j] = (unsigned char)res;
		}
	}

							
	for (i=1; i<sizeMinus; i++) { //common subexpression
		temp2 = i*SIZE;  //loop invariant 
		for ( j=1; j<sizeMinus; j++ ) { //common subexpression
		  	temp3 = temp2 + j;      //common subexpression
			t = pow((output[temp3] - golden[temp3]),2);
			PSNR += t;
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


int main(int argc, char* argv[])
{
	double PSNR;
	PSNR = sobel(input, output, golden);
	//printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	//printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");

	return 0;
}

