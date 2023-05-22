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
    
    int h_Histogram[256];//histogram gia thn gpu sto host
    
    int *d_Histogram = NULL;
    unsigned char *d_Input = NULL;
    unsigned int img_size = img_in.w * img_in.h;
    int *blocked_hists;
    
    int thread_reps;
    int block_num;
    int size_in_blocks = img_size / 256;//apo poses 256ades apoteleitai h eikona
    int which_kernel;
    
    int j = 0;
    int cdf = 0;
    int min = 0;
    int d = 0;
    int lut[256];
    
    int gpu_calc;//poso meros ths eikonas tha ypologisei h gpu
    
    //gia tis metriseis
    
    cudaStream_t stream;
    cudaEvent_t start_hist,end_hist;
    float hist_time;
    
    cudaEventCreate(&start_hist);
    cudaEventCreate(&end_hist);
    
    result.w = img_in.w;
    result.h = img_in.h;

    cudaHostAlloc((void **) &result.img, result.w * result.h * sizeof(unsigned char), cudaHostAllocDefault);
    
    printf("Number of blocks(256) of the image is: %d\n", size_in_blocks);
    printf("Enter how many blocks the gpu will compute : ");
    scanf("%d",&gpu_calc);
    
    if(gpu_calc > size_in_blocks){
        
        printf("Number of blocks should be less or equal than %d\n",gpu_calc);
        
        cudaFreeHost(result.img);
        cudaFreeHost(img_in.img);
        
        cudaDeviceReset();
        
        exit(1);
    }
    
    printf("Number of blocks for GPU is: %d and block size is 256...\n",gpu_calc);
    printf("Enter how many reps a thread will do (Enter a power of two): ");
    scanf("%d",&thread_reps);
    
    printf("Enter which kernel to run:(1-5) ");
    scanf("%d",&which_kernel);
    
    if(which_kernel > 5){
        
        printf("Pick a proper kernel no (1-5)\n");
        
        cudaFreeHost(result.img);
        cudaFreeHost(img_in.img);
        
        cudaDeviceReset();
        
        exit(1);
    }
    
    block_num = gpu_calc / thread_reps;
    
    if(block_num == 0){
        
        printf("Too few blocks or too many thread reps\n");
        
        cudaFreeHost(result.img);
        cudaFreeHost(img_in.img);
        
        cudaDeviceReset();
        
        exit(1);
    }
    
    cudaMalloc((void**)&d_Histogram,sizeof(int) * 256);
    cudaMalloc((void**)&d_Input,sizeof(unsigned char) * gpu_calc * 256);//na to allakso
    cudaMalloc((void**)&blocked_hists,sizeof(int) * 256 * block_num);
    
    if(d_Histogram == NULL || d_Input == NULL || blocked_hists == NULL){
        
        printf("Error in device memory allocation.Exiting...\n");
        
        cudaFreeHost(result.img);
        cudaFreeHost(img_in.img);
                
        cudaDeviceReset();
        exit(1);
    }

    //cudaMemset(blocked_hists, 0, sizeof(int) * 256 * block_num);
    cudaStreamCreate(&stream);

    cudaEventRecord(start_hist,stream);        
   
    
    cudaMemset(d_Histogram,0,256 * sizeof(int));
    
    cudaCheckError();
    
    //metafora eikonas
    
    cudaMemcpyAsync(d_Input, img_in.img ,gpu_calc * 256 * sizeof(unsigned char), cudaMemcpyHostToDevice,stream);//async meta

    cudaCheckError();
    
    //kernels
    
    if(which_kernel == 1){
        
        histogram_g1<<<block_num , 256, 0, stream>>>(d_Histogram ,d_Input,gpu_calc * 256);
        
        cudaCheckError();
        
    }else if(which_kernel == 2){

        cudaMemset(blocked_hists, 0, sizeof(int) * 256 * block_num);

        cudaCheckError();

        histogram_g2<<<block_num , 256, 0, stream>>>(d_Histogram , d_Input,blocked_hists,gpu_calc * 256);

        cudaCheckError();
        
    }else if(which_kernel == 3){
        
        histogram_s1<<<block_num , 256, 0, stream>>>(d_Histogram ,d_Input,gpu_calc * 256);
        
        cudaCheckError();
    
    }else if(which_kernel == 4){
        
        histogram_s3<<<block_num , 256, 0, stream>>>(d_Histogram ,d_Input,gpu_calc * 256);
        
        cudaCheckError();

    }else if(which_kernel == 5){
        
        cudaMemset(blocked_hists, 0, sizeof(int) * 256 * block_num);

        cudaCheckError();

        histogram_s4<<<block_num , 256, 0, stream>>>(d_Histogram ,d_Input,blocked_hists, gpu_calc * 256);

        cudaCheckError();
    
    }
    
    cudaMemcpyAsync(h_Histogram,d_Histogram,256* sizeof(int),cudaMemcpyDeviceToHost,stream);
    
    cudaCheckError();
    

    for(int i = 0; i < 256; i++)//isos na figei meta
        hist[i] = 0;
    
    //cpu computation
    
    for(int i = gpu_calc * 256; i < img_size; i++)
        hist[img_in.img[i]]++;
    
    cudaDeviceSynchronize();
    
    //merge gpu and cpu
    
    for(int i = 0; i < 256; i++)
        hist[i] += h_Histogram[i];
    
    /* ypologismos cdf sth cpu*/
    
    while(min == 0){
        
        lut[j] = 0;
        min = hist[j];
        
        j++;
    }
    
    d = img_in.w *img_in.h - min;
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
    
    result_calc_param<<<gpu_calc , 256, 0, stream>>>(d_Input,d_Histogram);//d_Result grafetai pano sto d_Input
    
    cudaCheckError();
    
    cudaMemcpyAsync(result.img,d_Input, 256 * gpu_calc * sizeof(unsigned char),cudaMemcpyDeviceToHost,stream);
    
    cudaCheckError();
    
    for(int i = gpu_calc * 256; i < img_size; i ++)
        result.img[i] = (unsigned char)lut[img_in.img[i]];
    
    cudaEventRecord(end_hist,stream);
    
    cudaEventSynchronize(end_hist);
    
    cudaEventElapsedTime(&hist_time,start_hist,end_hist);
    
    cudaCheckError();

    fprintf(stderr,"Total time of histogram(%d) = %lf\n",which_kernel,hist_time/1000);
	    
    cudaFree(d_Histogram);
    cudaFree(d_Input);
    cudaFree(blocked_hists);
    
    cudaEventDestroy(start_hist);
    cudaEventDestroy(end_hist);
    
    cudaStreamDestroy(stream);
    
    return result;
}
