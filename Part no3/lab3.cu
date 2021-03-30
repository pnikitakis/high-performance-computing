#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius = 16;

#define FILTER_LENGTH 	(2 * 16 + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.05 
#define TYPE float
#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

 //GPU code
 
 
 __constant__ TYPE d_Filter[FILTER_LENGTH];  

__global__ void convolROWS(TYPE* d_Buffer, 
		       TYPE* d_Input, 
		       int imageH, 
		       int imageW, 
		       int filterR)
{

      int k;
      int blockId = blockIdx.y * gridDim.x + blockIdx.x;
      int tx = threadIdx.y * blockDim.x + threadIdx.x;
      int threadId = blockId * (blockDim.x * blockDim.y) + tx;
      TYPE sum = 0;
      //tile = threadBlock = 1024
      __shared__ TYPE s_input[1024];
      
      //load to shared memory

      s_input[tx] = d_Input[threadId];
      __syncthreads();  // wait for all shared data to load
      
      //result
      for (k = -filterR; k <= filterR; k++) {
	int d = threadIdx.x + k;
	
	if (d >= 0 && (d < imageW)) {
          sum += s_input[tx + k] * d_Filter[filterR - k];
        }     

        d_Buffer[threadId] = sum;	
      }
      
      
      
}

__global__ void convolCOL(TYPE* d_Output_GPU, 
		       TYPE* d_Buffer, 
		       int imageH, 
		       int imageW, 
		       int filterR)
{

      int k;
      int blockId = blockIdx.y * gridDim.x + blockIdx.x;
      int ty = threadIdx.x * blockDim.y + threadIdx.y;

      TYPE sum = 0;
      //tile = threadBlock = 1024
      __shared__ TYPE s_input[1024];
      
    // load to shared memory
      
     s_input[ty] = d_Buffer[threadIdx.y*imageW + blockId*blockDim.x + threadIdx.x];
      __syncthreads(); // wait for all shared data to load
      
	// result
      for (k = -filterR; k <= filterR; k++) {
	int d = threadIdx.y + k;
	
	if (d >= 0 && (d < imageW)) {
          sum += s_input[ty + k] * d_Filter[filterR - k];
        }     

        d_Output_GPU[threadIdx.y*imageW + blockId*blockDim.x + threadIdx.x] = sum;	
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
    *d_Output_GPU,
    *d_Buffer;
    
    int imageW;
    int imageH;
    unsigned int i;

    //printf("Enter filter radius : ");
    //scanf("%d", &filter_radius);
    //filter_radius = 16;
	
    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

 //   printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
  //  printf("Allocating and initializing host arrays...\n");

    h_Filter    = (TYPE *)malloc(FILTER_LENGTH * sizeof(TYPE));
    h_Input     = (TYPE *)malloc(imageW * imageH * sizeof(TYPE));
    h_Buffer    = (TYPE *)malloc(imageW * imageH * sizeof(TYPE));
    h_OutputCPU = (TYPE *)malloc(imageW * imageH * sizeof(TYPE));
    h_OutputGPU = (TYPE *)malloc(imageW * imageH * sizeof(TYPE));     
    
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
/*
    printf("CPU computation...\n");

    clock_t begin = clock();
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); 
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); 
    clock_t end = clock();
    double cpu_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Cpu time: %lf\n",cpu_time);
    */
    cudaMalloc((void**)&d_Input, imageW * imageH * sizeof(TYPE));
    cudaMalloc((void**)&d_Output_GPU,  imageW * imageH * sizeof(TYPE));
    cudaMalloc((void**)&d_Buffer,  imageW * imageH * sizeof(TYPE));

    if(!(d_Input || d_Output_GPU || d_Buffer)){
	printf("Malloc allocation problem on device, exiting.. \n");
	return(1);
    }
    
    
    //TIME 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float tot_time=0, timer = 0;
    
    
    
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(TYPE), cudaMemcpyHostToDevice);
        // d_Filter is in __constant__ memory
    cudaMemcpyToSymbol(d_Filter, h_Filter, sizeof(TYPE) * FILTER_LENGTH);
    
    cudaCheckError(); 
    
    //block & grid dimensions
    int x_block, y_block, x_grid, y_grid;

    
    if(imageW < 1024){
	x_block = imageW;
	y_block = 1024 / imageW;
	
	x_grid = 1;
	y_grid = (imageW*imageW)/1024; 
    } 
    else{
      if( imageW == 1024){
	x_block = 1024;
	y_block = 1;
	
	x_grid = 1;
	y_grid = imageW;
      }
      else{
	x_block = 1024;
	y_block = 1;
	
	x_grid = (imageW*imageW)/1024;
	y_grid = imageH;
      }
    }
     
    dim3 grid(x_grid , y_grid);  
    dim3 block(x_block, y_block);

    
    cudaEventRecord(start);
    convolROWS<<<grid , block>>>(d_Buffer, d_Input, imageH, imageW, filter_radius);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer, start, stop);
    printf("Kernel time for rows: %f\n", timer);
    tot_time += timer;
    
    //wait 1st kernel to finish
    cudaThreadSynchronize();
    //check for errors
    cudaCheckError();
    

    dim3 grid2(y_grid , x_grid);  
    dim3 block2(y_block, x_block);
    
    cudaEventRecord(start);
    convolCOL<<<grid2 , block2>>>(d_Output_GPU, d_Buffer, imageH, imageW, filter_radius);
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
    cudaMemcpy(h_OutputGPU, d_Output_GPU, imageW * imageH * sizeof(TYPE), cudaMemcpyDeviceToHost);
    
    printf("Total time for GPU: %f\n", tot_time);
    
    // comment out CPU GPC comparison code since it is correct
   
    //compare 
  /*  
    TYPE max = 0.0;
    unsigned int counter=0;
    
     for (i = 0; i < imageW * imageH; i++) {
	if(ABS(h_OutputGPU[i] - h_OutputCPU[i]) > max){
	  max = ABS(h_OutputGPU[i] - h_OutputCPU[i]);

	  printf("iteration= %d max=%10g \n",i, max);
	  counter++;
	}
	
     }
     printf("for %d filter, max= %d, counter=%d\n", filter_radius, max, counter);
     
     */
     
    
    // free all the allocated memory
    free(h_OutputCPU);
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    cudaFree(d_Input);
    cudaFree(d_Output_GPU);

    cudaDeviceReset();


    return 0;
}
