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


unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];




double sobel(unsigned char *input, unsigned char *output, unsigned char *golden)
{
	double PSNR = 0, t;
	int i, j, k, l;
	unsigned int p;
	int res, resX, resY;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;

	
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < SIZE-1; i++) {
		output[i*SIZE] = 0;
		output[i*SIZE + SIZE - 1] = 0;
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

	fread(input, sizeof(unsigned char), SIZE*SIZE, f_in);
	fread(golden, sizeof(unsigned char), SIZE*SIZE, f_golden);
	fclose(f_in);
	fclose(f_golden);
  
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

	for (j=1; j<SIZE-1; j+=1) {
		for (i=1; i<SIZE-1; i+=1 ) {
			
			
			/* START first convolution2D() for horizontial */
			res = 0;  
			
			for (k = -1; k <= 1; k++) {
				for (l = -1; l <= 1; l++) {
					res += input[(i + l)*SIZE + j + k] * horiz_operator[l+1][k+1];
				}
			}					
			
			resX = pow(res,2); 
			/* END first convolution2D() for horizontial */
				
			/* START first convolution2D() for vertical */
			res=0; 
			
			for (k = -1; k <= 1; k++) { 
				for (l = -1; l <= 1; l++) {
					res += input[(i + l)*SIZE + j + k] * vert_operator[l+1][k+1];
				}
			}	
			
			resY = pow(res,2);
			/* END first convolution2D() for vertical */
			
			
			res = (int)sqrt(resX + resY);
			
			if (res > 255)
				output[i*SIZE + j] = 255;      
			else
				output[i*SIZE + j] = (unsigned char)res;
		}
	}

							
	for (i=1; i<SIZE-1; i++) {
		for ( j=1; j<SIZE-1; j++ ) {
			t = pow((output[i*SIZE+j] - golden[i*SIZE+j]),2);
			PSNR += t;
		}
	}
  
	PSNR /= (double)(SIZE*SIZE);
	PSNR = 10*log10(65536/PSNR);

	
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("%10g\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

  
	fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
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

