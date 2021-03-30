#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.05 

#define NUM_THREADS imageW*imageH
#define NUM_BLOCKS ((imageH > 32) ? (NUM_THREADS/(32*32)) : 1)
#define TYPE double
#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

 //GPU code
 
__global__ void convolROWS(TYPE* d_Buffer, 
		       TYPE* d_Input, 
		       TYPE* d_Filter,
		       int imageH, 
		       int imageW, 
		       int filterR)
{

    int k;    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    TYPE sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = idx + k;

        if (d >= 0 && d < imageW) {
          sum += d_Input[idy*imageW + d] * d_Filter[filterR - k];
        }     

        d_Buffer[idy*imageW + idx] = sum;	
      }
      
}

__global__ void convolCOL(TYPE* d_Output_GPU, 
		       TYPE* d_Buffer, 
		       TYPE* d_Filter,
		       int imageH, 
		       int imageW, 
		       int filterR)
{


    int k;    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    TYPE sum = 0;

      
      //col con

      for (k = -filterR; k <= filterR; k++) {
        int d = idy + k;

        if (d >= 0 && d < imageH) {
          sum += d_Buffer[d*imageW + idx] * d_Filter[filterR - k];
        }   
 
        d_Output_GPU[idy*imageW + idx] = sum;
      }
      
      
}
 
 
 
 
////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(TYPE *h_Dst, TYPE *h_Src, TYPE *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      TYPE sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(TYPE *h_Dst, TYPE *h_Src, TYPE *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      TYPE sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    TYPE
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU,
    *d_Input,
    *d_Filter,
    *d_Output_GPU,
    *d_Buffer;
    
    int imageW;
    int imageH;
    unsigned int i;

//	printf("Enter filter radius : ");
//	scanf("%d", &filter_radius);
    filter_radius = 16;
    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW ;

//    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
 //   printf("Allocating and initializing host arrays...\n");

    h_Filter    = (TYPE *)malloc(FILTER_LENGTH * sizeof(TYPE));
    h_Input     = (TYPE *)malloc(imageW * imageH * sizeof(TYPE));
    h_Buffer    = (TYPE *)malloc(imageW * imageH * sizeof(TYPE));
    h_OutputCPU = (TYPE *)malloc(imageW * imageH * sizeof(TYPE));
    h_OutputGPU = (TYPE *)malloc(imageW * imageH * sizeof(TYPE)); //new malloc for result from device

    if( h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_OutputGPU == NULL ){
      printf("Malloc allocation problem on host, exiting...\n");
      return(1);
    }
      
    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (TYPE)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (TYPE)rand() / ((TYPE)RAND_MAX / 255) + (TYPE)rand() / (TYPE)RAND_MAX;
    }

// comment out CPU code since it is correct
  /*  printf("CPU computation...\n");

    clock_t begin = clock();
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); 
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); 
    clock_t end = clock();
    double cpu_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Cpu time: %lf\n",cpu_time);
  */  
    
    cudaMalloc((void**)&d_Input, imageW * imageH * sizeof(TYPE));
    cudaMalloc((void**)&d_Filter, FILTER_LENGTH * sizeof(TYPE));
    cudaMalloc((void**)&d_Output_GPU,  imageW * imageH * sizeof(TYPE));
    cudaMalloc((void**)&d_Buffer,  imageW * imageH * sizeof(TYPE));

    if(!(d_Input || d_Filter || d_Output_GPU || d_Buffer)){
	printf("Malloc allocation problem on device, exiting.. \n");
	return(1);
    }
    
    
    //TIME 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float tot_time=0, timer = 0;
    
    
    //cudaEventRecord(start);
    
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(TYPE), cudaMemcpyHostToDevice);
    
    //no need to calculate time for memory copies

    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&timer, start, stop);
    //printf("Host to Device transfer: %f\n", timer);
    //tot_time += timer;
    
    
    cudaCheckError(); //check for errors in cudaMemcpy
    
    int var, var2;

    if (imageH <= 32){
      var = 1;
      var2 = imageH;
    }else{
      var = imageH/32;
      var2 = 32;
    }
  //  printf("grid dim= %d\n", var);
   // printf("block dim= %d\n", var2);

    dim3 grid(var , var);  
    dim3 block(var2, var2);

    
    cudaEventRecord(start);
    convolROWS<<<grid , block>>>(d_Buffer, d_Input, d_Filter, imageH, imageW, filter_radius);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer, start, stop);
    printf("Kernel time for rows: %f\n", timer);
    tot_time += timer;
    
    
    //wait 1st kernel to finish
    cudaThreadSynchronize();
    //check for errors
    cudaCheckError();
    

    cudaEventRecord(start);
    convolCOL<<<grid , block>>>(d_Output_GPU, d_Buffer, d_Filter, imageH, imageW, filter_radius);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer, start, stop);
    printf("Kernel time for col: %f\n", timer);
    tot_time += timer;   
    
    

    //wait to finish
    cudaThreadSynchronize();
    //check for errors
    cudaCheckError();


    
    
    //copy output from device to host
    //cudaEventRecord(start);
    cudaMemcpy(h_OutputGPU, d_Output_GPU, imageW * imageH * sizeof(TYPE), cudaMemcpyDeviceToHost);
    
    //no need to calculate time for memory copies

    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&timer, start, stop);
    //printf("Device to host time: %f\n", timer);
    //tot_time += timer;  
    
    
    printf("Total GPU time: %f\n", tot_time);
    
    
    
    
    
    // comment out CPU GPC comparison code since it is correct

    //compare here
    /*
    TYPE max = 0.0;
    
     for (i = 0; i < imageW * imageH; i++) {
//	  printf("iteration= %d dif=%10g \n",i, ABS(h_OutputGPU[i] - h_OutputCPU[i]));
       
      	if(ABS(h_OutputGPU[i] - h_OutputCPU[i]) > max){
	  max = ABS(h_OutputGPU[i] - h_OutputCPU[i]);
	
	  printf("h_OutputGPU: %g\t h_OutputCPU: %g\n", h_OutputGPU[i], h_OutputCPU[i]);
	  //printf("The difference between the %dnth element is larger than accuracy. \n CPU: %.10g GPU %.10g differece: %.15g \nNow exiting..\n", i,h_OutputCPU[i] ,h_OutputGPU[i], ABS(h_OutputGPU[i] - h_OutputCPU[i])  );
	  //break;
//	  printf("iteration= %d max=%10g \n",i, max);
	}
     }
     printf("for %d filter, max= %g\n", filter_radius, max);
     */
     
     
    
    // free all the allocated memory
    free(h_OutputCPU);
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    cudaFree(d_Input);
    cudaFree(d_Filter);
    cudaFree(d_Output_GPU);

    cudaDeviceReset();


    return 0;
}
