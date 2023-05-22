#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <time.h>

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
            cudaFree(d_Histogram);\
            cudaFree(d_Input);\
            cudaFreeHost(result.img);\
            cudaFree(blocked_hists);\
            cudaFreeHost(img_in.img);\
            cudaDeviceReset();\
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    unsigned int img_size = img_in.w * img_in.h;
    
    int *d_Histogram = NULL;
    unsigned char *d_Input = NULL;
    int *blocked_hists = NULL;
    
    int block_num;
    int size_in_blocks = img_size / 256;//apo poses 256ades apoteleitai h eikona
    
    int j = 0;
    int cdf = 0;
    int min = 0;
    int d = 0;
    int lut[256];
    

    
    cudaStream_t stream;
    cudaEvent_t start_hist,end_hist;
    float hist_time;
    

    
    cudaEventCreate(&start_hist);
    cudaEventCreate(&end_hist);
    
    result.w = img_in.w;
    result.h = img_in.h;

    cudaHostAlloc((void **) &result.img, img_size * sizeof(unsigned char), cudaHostAllocDefault);

   
    block_num = size_in_blocks / 32;//32 reps(h 33...analoga to ypoloipo) kathe thread
    
    if(block_num == 0){
        
        printf("Image too small\n");
        
        exit(1);
    }
    
    cudaMalloc((void**)&d_Histogram,sizeof(int) * 256);
    cudaMalloc((void**)&d_Input,sizeof(unsigned char) * img_size);//olo sth gpu?
    cudaMalloc((void**)&blocked_hists,sizeof(int) * 256 * block_num);
    
    if(d_Histogram == NULL || d_Input == NULL || blocked_hists == NULL){
        
        printf("Error in device memory allocation.Exiting...\n");
        cudaDeviceReset();
        exit(1);
    }

    cudaStreamCreate(&stream);

    cudaEventRecord(start_hist,stream);        
    
    cudaMemsetAsync(d_Histogram,0,256 * sizeof(int));
    
    cudaCheckError();
    
    cudaMemsetAsync(blocked_hists, 0, sizeof(int) * 256 * block_num);

    cudaCheckError();
    
    //metafora eikonas
    
    cudaMemcpyAsync(d_Input, img_in.img ,img_size * sizeof(unsigned char), cudaMemcpyHostToDevice,stream);

    cudaCheckError();
    
    //kernel
    
    histogram_g2<<<block_num , 256, 0, stream>>>(d_Histogram , d_Input,blocked_hists,img_size);

    cudaCheckError();
    
    //metafora histogram sth cpu
    
    cudaMemcpyAsync(hist,d_Histogram,256* sizeof(int),cudaMemcpyDeviceToHost,stream);
    
    cudaCheckError();
    
    cudaDeviceSynchronize();
        
    /* ypologismos cdf */
    
    while(min == 0){
        
        lut[j] = 0;
        min = hist[j];
        
        j++;
    }
    
    d = img_size - min;
    j--;
    
    for(; j < 256; j++){
        
        cdf += hist[j];

        lut[j] = (int)(((float)cdf - min)*255/d + 0.5);
        
        if(lut[j] > 255)
            lut[j] = 255;
        
        if(lut[j] < 0)
            lut[j] = 0;
    }
    
    cudaMemcpyAsync(d_Histogram,lut,256 * sizeof(int),cudaMemcpyHostToDevice,stream);//metaferontai sthn d_Histogram gia na mhn ksanadesmeyseis mnhmh
    
    cudaCheckError();
    
    result_calc<<<size_in_blocks / 32, 256, 0, stream>>>(d_Input,d_Histogram,img_size);
    
    cudaCheckError();
    
    cudaMemcpyAsync(result.img,d_Input, img_size,cudaMemcpyDeviceToHost,stream);
    
    cudaCheckError();
    
    cudaEventRecord(end_hist,stream);
    
    cudaEventSynchronize(end_hist);
    
    cudaEventElapsedTime(&hist_time,start_hist,end_hist);
    
    cudaCheckError();

    fprintf(stderr,"Total time = %lf\n",hist_time/1000);
	    
    cudaFree(d_Histogram);
    cudaFree(d_Input);
    cudaFree(blocked_hists);
    
    cudaEventDestroy(start_hist);
    cudaEventDestroy(end_hist);
    
    cudaStreamDestroy(stream);
    
    return result;
}
