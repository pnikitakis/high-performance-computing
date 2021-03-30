#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "hist-equ.h"

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }


__global__ void hist(unsigned char * img_in, int img_size, int * tot_his){

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int tile_w = blockDim.x * gridDim.x; //tile = total threads

	//1 thread gia 1 thesh tou topikou pinaka histogram
	__shared__ int s_his[256];

	// arxikopoihsh histogram sthn shared mem
	s_his[threadIdx.x] = 0;
	
	__syncthreads();
	for(int p=0; p < img_size ; p+= tile_w){

		//gia to teleutaio tile poy tha perisseuei kommati
		if(p + tx > img_size)
			break;
		
		//kateu8eian apo global read kai meta atomic add sthn shared
		atomicAdd(&s_his[img_in[p + tx]], 1);
		//s_tile[tx+1,2,3] gia bank conf ? na alaksw tile ?

	}	
	__syncthreads();//wait na gemisoun ta s_his
	
	//ekasto thread grafei to shared histogram sto global 
	atomicAdd(&(tot_his[threadIdx.x]), s_his[threadIdx.x]);

	
}

__constant__ int d_lut[256]; //constant giati den allazei kai to diabazoume sunexeia

__global__ void res_img(unsigned char * img_in, int img_size, unsigned char* img_out){

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int tile_w = blockDim.x * gridDim.x; //tile = total threads

	for(int p=0; p < img_size; p += tile_w){
	  
	    if(p + tx > img_size)
	      break;
	
	    img_out[p + tx] = d_lut[img_in[p + tx]];
	    
	}


}




__device__ unsigned char * d_img_in; //global device gia na diabastoun 1 fora apo thn GPU
__device__ int * d_img_size;

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){

	int * d_hist_out;
	int * d_img_size_h;
	unsigned char * d_img_in_h;

	cudaMalloc((void**)&d_img_in_h, img_size*sizeof(unsigned char));	
	cudaMalloc((void**)&d_img_size_h, sizeof(int));
	cudaMalloc((void**)&d_hist_out, nbr_bin*sizeof(int));

	if(!(d_img_in_h || d_hist_out || d_img_size_h)){
		printf("Malloc allocation problem on device, exiting.. \n");
		exit(1);
	}

	cudaGetSymbolAddress((void **)&d_img_in_h, "d_img_in") ;
	cudaGetSymbolAddress((void **)&d_img_size_h, "d_img_size") ;
	cudaMemset(d_hist_out, 0, nbr_bin*sizeof(int)); //arxikopoihsh me 0 to histogram
	cudaMemcpy(d_img_in_h, img_in, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_img_size_h, &img_size, sizeof(int), cudaMemcpyHostToDevice);

	cudaCheckError();


	   //TIME 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float tot_time=0, timer = 0;
    
    cudaEventRecord(start);
    hist<<<64, 256>>>(d_img_in_h, *d_img_size_h, d_hist_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer, start, stop);
    printf("Kernel time for histogram: %f\n", timer);
    tot_time += timer;

     //wait 1st kernel to finish
    cudaThreadSynchronize();
    //check for errors
    cudaCheckError();
    
    
    cudaMemcpy(hist_out, d_hist_out, nbr_bin * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(d_hist_out);
   //cudaFree(d_img_in);
    cudaFree(&d_img_size);
    
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d; 

    cdf = 0;
    min = 0;
    i = 0;
// algorithm gia edw | extra J gia th thesh ?
    while(min == 0){
        min = hist_in[i++];
    }

    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
	// gia i<j cdf(j) = cdfMIN -> lut[i]=0
        if(lut[i] < 0){
            lut[i] = 0;
        }else
	{
	if(lut[i] > 255)
		lut[i] = 255;
	}
        
        
    }


    //second kernel
    
    //lut sthn constant mem
    cudaMemcpyToSymbol(d_lut, lut, nbr_bin*sizeof(int));
    
    unsigned char * d_img_out;
    int * d_img_size_h;
    unsigned char * d_img_in_h;
    
    cudaGetSymbolAddress((void **)&d_img_in_h, "d_img_in") ;
    cudaGetSymbolAddress((void **)&d_img_size_h, "d_img_size") ;
	
	
    if(!cudaMalloc((void**)&d_img_out, img_size*sizeof(unsigned char))){
      printf("Malloc allocation problem on device, exiting.. \n");
      exit(1);
    }
      
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float tot_time=0, timer = 0;
    
    cudaEventRecord(start);
    res_img<<<64, 256>>>(d_img_in_h, *d_img_size_h, d_img_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer, start, stop);
    printf("Kernel time for lut: %f\n", timer);
    tot_time += timer;

     //wait 2st kernel to finish
    cudaThreadSynchronize();
    //check for errors
    cudaCheckError();
    
    
    cudaMemcpy(img_out, d_img_out, img_size* sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaCheckError();

    cudaFree(d_img_out);
    cudaFree(d_img_size_h);
    cudaFree(d_img_in_h);
    
    
    
}
