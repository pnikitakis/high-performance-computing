#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.05 

#define NUM_THREADS imageW*imageH
#define NUM_BLOCKS ((imageH > 32) ? (NUM_THREADS/(32*32)) : 1)

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

 //GPU code
 
__global__ void convolROWS(float* d_Buffer, 
		       float* d_Input, 
		       float* d_Filter,
		       int imageH, 
		       int imageW, 
		       int filterR)
{

    int k;    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = idx + filterR + k;

        sum += d_Input[(idy+filterR)*(imageW+2*filterR) + d] * d_Filter[filterR - k];
             

        d_Buffer[(idy+filterR)*(imageW+2*filterR) + idx + filterR] = sum;	
      }
      
}

__global__ void convolCOL(float* d_Output_GPU, 
		       float* d_Buffer, 
		       float* d_Filter,
		       int imageH, 
		       int imageW, 
		       int filterR)
{


    int k;    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0;

      
      //col con

      for (k = -filterR; k <= filterR; k++) {
        int d = idy + filterR + k;

        sum += d_Buffer[d*(imageW+2*filterR) + idx + filterR] * d_Filter[filterR - k];
           
 
        d_Output_GPU[(idy+filterR)*(imageW +2*filterR)+ idx+filterR] = sum;
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
    *d_Buffer,
    *h_newInput,
    *d_newInput,
    *h_tempOutputGPU;
    
    
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
    h_Buffer    = (float *)malloc((imageW * imageH + 2*filter_radius*(imageW + 2*filter_radius) + 2*filter_radius*imageH )* sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc((imageW * imageH + 2*filter_radius*(imageW + 2*filter_radius) + 2*filter_radius*imageH )* sizeof(float));
    h_newInput = (float *)malloc((imageW * imageH + 2*filter_radius*(imageW + 2*filter_radius) + 2*filter_radius*imageH )* sizeof(float));
    h_tempOutputGPU = (float *)malloc((imageW * imageH + 2*filter_radius*(imageW + 2*filter_radius) + 2*filter_radius*imageH )* sizeof(float));
    
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
    cudaMalloc((void**)&d_Output_GPU,  ((imageW * imageH + 2*filter_radius*(imageW + 2*filter_radius) + 2*filter_radius*imageH )* sizeof(float)));
    cudaMalloc((void**)&d_Buffer, ((imageW * imageH + 2*filter_radius*(imageW + 2*filter_radius) + 2*filter_radius*imageH )* sizeof(float)));
    cudaMalloc((void**)&d_newInput, ((imageW * imageH + 2*filter_radius*(imageW + 2*filter_radius) + 2*filter_radius*imageH )* sizeof(float)));

    if(!(d_Input || d_Filter || d_Output_GPU || d_Buffer)){
	printf("Malloc allocation problem on device, exiting.. \n");
	return(1);
    }
    
    
    //peirazw h_Input + d_Input
    
    int met = 0, met2 = 0, met3=0;
    
    //mhdenismos R arxikwn grammwn
    for(met=0; met < filter_radius*(imageW + 2*filter_radius); met++ ){
      h_newInput[met] = 0.0;
    }
    
    //mhdenismos R telikwn grammwn
    for(met= ((2*filter_radius+ imageW)*filter_radius + (2*filter_radius+ imageW)*imageH ); met < (imageW * imageH + 2*filter_radius*(imageW + 2*filter_radius) + 2*filter_radius*imageH); met++ ){
      h_newInput[met] = 0.0;
    }
    
    //elegxos!!!!
    for(met=filter_radius ; met < imageH + filter_radius; met++){
      //kanonika data
      memcpy(&h_newInput[met*(2*filter_radius + imageW) + filter_radius], &h_Input[met3*imageW], sizeof(float)*imageW);
      //mhdenismos akrwn
	for(met2=0; met2 < filter_radius; met2++){
	   h_newInput[met*(2*filter_radius + imageW) + met2] = 0.0;
	   h_newInput[met*(2*filter_radius + imageW) + imageW + met2] = 0.0;
	}
	met3 += 1;
    }
    
    
    
    cudaMemcpy(d_newInput, h_newInput, ((imageW * imageH + 2*filter_radius*(imageW + 2*filter_radius) + 2*filter_radius*imageH )* sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaCheckError(); //check for errors in cudaMemcpy
    
    int var, var2;

    if (imageH <= 32){
      var = 1;
      var2 = imageH;
    }else{
      var = imageH/32;
      var2 = 32;
    }
    printf("grid dim= %d\n", var);
    printf("block dim= %d\n", var2);

    dim3 grid(var , var);  
    dim3 block(var2, var2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
     float tot_time=0, timer = 0;
     
 
    cudaEventRecord(start);

    convolROWS<<<grid , block>>>(d_Buffer, d_newInput, d_Filter, imageH, imageW, filter_radius);
      
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer, start, stop);
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
    tot_time += timer;
    
    printf("Total time for both kernels: %lf\n\n", tot_time);
    
    
    //wait to finish
    cudaThreadSynchronize();
    //check for errors
    cudaCheckError();

    
    
    
    //copy output from device to host
    cudaMemcpy(h_tempOutputGPU, d_Output_GPU, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);

    
    //h_OutputGPU
    for(met=0; met < imageH; met++){
      memcpy(&h_OutputGPU[met*imageW] , &h_tempOutputGPU[(2*filter_radius+ imageW)*filter_radius + met*(2*filter_radius+ imageW) + filter_radius ],sizeof(float)*imageW);
    }
    //compare here
    
  
     
     
    
    // free all the allocated memory
    free(h_OutputCPU);
    free(h_OutputGPU);
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
