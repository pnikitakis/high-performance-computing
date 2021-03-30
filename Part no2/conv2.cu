#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.05 

#define NUM_THREADS imageW*imageH
#define NUM_BLOCKS 1

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

 //GPU code
 
__global__ void convolROWS(float* d_Output_GPU, 
		       float* d_Input, 
		       float* d_Filter,
		       int imageH, 
		       int imageW, 
		       int filterR)
{

  int k, i = blockIdx.x *blockDim.x + threadIdx.x;

  
  //row conv
  float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = i%imageW + k;

        if (d >= 0 && d < imageW) {
          sum += d_Input[i + k] * d_Filter[filterR - k];
        }     

        d_Output_GPU[i] = sum;	
      }
      
}

__global__ void convolCOL(float* d_Output_GPU, 
		       float* d_Input, 
		       float* d_Filter,
		       int imageH, 
		       int imageW, 
		       int filterR)
{

  int k, i = blockIdx.x *blockDim.x + threadIdx.x;


      
      //col conv
  float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = i/imageW + k;

        if (d >= 0 && d < imageH) {
          sum += d_Input[d*imageW + i%imageW] * d_Filter[filterR - k];
        }   
 
        d_Output_GPU[i] = sum;
      }
      
      
}
 
 
 
 
////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

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
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

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
    
    float
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

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");

    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float)); //new malloc for result from device

    if( h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_OutputGPU == NULL ){
      printf("Malloc allocation problem on host, exiting...\n");
      return(1);
    }
      
    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }


    printf("CPU computation...\n");

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); 
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); 


    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

    cudaMalloc((void**)&d_Input, imageW * imageH * sizeof(float));
    cudaMalloc((void**)&d_Filter, FILTER_LENGTH * sizeof(float));
    cudaMalloc((void**)&d_Output_GPU,  imageW * imageH * sizeof(float));
    cudaMalloc((void**)&d_Buffer,  imageW * imageH * sizeof(float));

    if(!(d_Input || d_Filter || d_Output_GPU || d_Buffer)){
	printf("Malloc allocation problem on device, exiting.. \n");
	return(1);
    }
    
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaCheckError(); //check for errors in cudaMemcpy
    
    convolROWS<<<NUM_BLOCKS , NUM_THREADS>>>(d_Buffer, d_Input, d_Filter, imageH, imageW, filter_radius);
     
    //wait 1st kernel to finish
    cudaThreadSynchronize();
    //check for errors
    cudaCheckError();
    
    convolCOL<<<NUM_BLOCKS , NUM_THREADS>>>(d_Output_GPU, d_Buffer, d_Filter, imageH, imageW, filter_radius);
    
    //wait to finish
    cudaThreadSynchronize();
    //check for errors
    cudaCheckError();

    
    
    
    //copy output from device to host
    cudaMemcpy(h_OutputGPU, d_Output_GPU, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);

    //compare here
    
    
     for (i = 0; i < imageW * imageH; i++) {
      	if(ABS(h_OutputGPU[i] - h_OutputCPU[i]) > accuracy){
	  printf("The difference between the %dnth element is larger than accuracy. \n CPU: %g GPU %g differece: %.15g \nNow exiting..\n", i,h_OutputCPU[i] ,h_OutputGPU[i], ABS(h_OutputGPU[i] - h_OutputCPU[i])  );
	  break;
	}
     }
     
     
     
     
    
    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    cudaFree(d_Input);
    cudaFree(d_Filter);
    cudaFree(d_Output_GPU);
    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
