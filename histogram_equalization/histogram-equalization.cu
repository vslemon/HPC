#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

//TO PIO APLO:
//KATHE THREAD DIAVAZEI ENA(h kai perissotera) STOIXEIO KAI GRAFEI TO APOTELESMA STO ANTISTOIXO STOIXEIO TOU global_histogram

__global__ void histogram_g1(int * hist_out , unsigned char * img_in, int size_of_img){
    
    int index = threadIdx.x + blockDim.x * blockIdx.x;//deixnei poio stoixeio tha parei apth mnhmh
    
    for(; index < size_of_img; index += gridDim.x * blockDim.x)
        atomicAdd(&hist_out[img_in[index]],1);
}

//KATHE THREAD DIAVAZEI ENA STOIXEIO(h kai perissotera) KAI GRAFEI TO APOTELESMA STO block_histogram.META KATHE THREAD PAIRNEI TO ANTISTOIXO STOIXEIO APTO
//block_histogram KAI TO APOTHIKEYEI STO global_histogram

__global__ void histogram_g2(int * hist_out, unsigned char * img_in,int *block_histogram,int size_of_img){
    
    int index = threadIdx.x + blockDim.x * blockIdx.x;//deixnei poio stoixeio tha parei apth mnhmh
    int current_block_histogram = blockIdx.x * 256;//block_histogram einai pinakas 256 * num_of_blocks.opote current_block_histogram deixnei sto antistoixo block
    
    for(; index < size_of_img; index += gridDim.x * blockDim.x)
        atomicAdd(&block_histogram[current_block_histogram + img_in[index]],1);
    
    __syncthreads();
    
    if(block_histogram[current_block_histogram + threadIdx.x])
        atomicAdd(&hist_out[threadIdx.x],block_histogram[current_block_histogram + threadIdx.x]);//na to ksanado
}

/*TA IDIA ALLA ME SHARED*/

__global__ void histogram_s1(int * hist_out, unsigned char * img_in, int size_of_img){
    
    int index = threadIdx.x + blockDim.x * blockIdx.x;//deixnei poio stoixeio tha parei apth mnhmh
    
    __shared__ int block_histogram[256];
    
    //kane 0 to block_histogram
    block_histogram[threadIdx.x] = 0;
    
    __syncthreads();
    
    for(; index < size_of_img; index += gridDim.x * blockDim.x)
        atomicAdd(&block_histogram[img_in[index]],1);
    
    __syncthreads();
    
    if(block_histogram[threadIdx.x])
       atomicAdd(&hist_out[threadIdx.x],block_histogram[threadIdx.x]);//na to ksanado
}

/*warp - aggregation*/

__global__ void histogram_s3(int * hist_out , unsigned char * img_in , int size_of_img){
    
    __shared__ int block_histogram[256];
    
    unsigned int peers = 0;
    short int is_peer = 0;
    
    //All lanes are available
    unsigned int unclaimed = 0xffffffff;//set all 32 bits
    
    int index = threadIdx.x + blockDim.x * blockIdx.x;//deixnei poio stoixeio tha parei apth mnhmh
    int element_of_in;
    int lane_num = threadIdx.x % 32;//arithos tou thread mesa sto warp
    
    block_histogram[threadIdx.x] = 0;
    
    __syncthreads();
    
    for(; index < size_of_img; index += gridDim.x * blockDim.x){
        
        element_of_in = img_in[index];
        
        do {
            // fetch key of first unclaimed lane and compare with this key
            is_peer = (element_of_in == __shfl_sync(unclaimed,element_of_in, __ffs(unclaimed) - 1));

            // determine which lanes had a match
            peers = __ballot_sync(unclaimed,is_peer);

            // remove lanes with matching keys from the pool
            unclaimed ^= peers;

        // quit if we had a match
        }while (!is_peer);
        
        if(lane_num == __ffs(peers) - 1)
            atomicAdd(&block_histogram[element_of_in], __popc(peers)); 
        
        unclaimed = 0xffffffff;
        
        __syncwarp();
    }
    
    __syncthreads();
    
    if(block_histogram[threadIdx.x])
       atomicAdd(&hist_out[threadIdx.x],block_histogram[threadIdx.x]);
}


//kai vale memset mesa sto if

__global__ void histogram_s4(int * hist_out , unsigned char * img_in ,int * block_histogram, int size_of_img){
    
    int current_block_histogram = blockIdx.x * 256;
    
    unsigned int peers = 0;
    short int is_peer = 0;
    
    //All lanes are available
    unsigned int unclaimed = 0xffffffff;//set all 32 bits
    
    int index = threadIdx.x + blockDim.x * blockIdx.x;//deixnei poio stoixeio tha parei apth mnhmh
    int element_of_in;
    int lane_num = threadIdx.x % 32;//arithos tou thread mesa sto warp
    
    for(; index < size_of_img; index += gridDim.x * blockDim.x){
        
        element_of_in = img_in[index];
        
        do {
            // fetch key of first unclaimed lane and compare with this key
            is_peer = (element_of_in == __shfl_sync(unclaimed,element_of_in, __ffs(unclaimed) - 1));

            // determine which lanes had a match
            peers = __ballot_sync(unclaimed,is_peer);

            // remove lanes with matching keys from the pool
            unclaimed ^= peers;

        // quit if we had a match
        }while (!is_peer);
        
        if(lane_num == __ffs(peers) - 1)
            atomicAdd(&block_histogram[current_block_histogram + element_of_in],__popc(peers));
        
        unclaimed = 0xffffffff;
        
        __syncwarp();
    }
    
    __syncthreads();
    
    if(block_histogram[current_block_histogram + threadIdx.x])
       atomicAdd(&hist_out[threadIdx.x],block_histogram[current_block_histogram + threadIdx.x]);
} 


__global__ void result_calc(unsigned char * img_out,int * lut,unsigned int img_size){//d_Input kai result vriskontai sthn idia mnhmh gia eksikonomisi xoroy

    __shared__ int shared_lut[256];
    int index =  threadIdx.x + blockDim.x * blockIdx.x;
    
    shared_lut[threadIdx.x] = lut[threadIdx.x];

    __syncthreads();
    
    for(; index < img_size; index += gridDim.x * blockDim.x)
        img_out[index] = (unsigned char)shared_lut[img_out[index]];//sigoureyoume sth cpu oti 0 <= lut[img_in[index]] <= 255 
}

__global__ void result_calc_param(unsigned char * img_out,int * lut){//d_Input kai result vriskontai sthn idia mnhmh gia eksikonomisi xoroy

    __shared__ int shared_lut[256];

    shared_lut[threadIdx.x] = lut[threadIdx.x];

    __syncthreads();

    int index =  threadIdx.x + blockDim.x * blockIdx.x;
    
    img_out[index] = (unsigned char)shared_lut[img_out[index]];//sigoureyoume sth cpu oti 0 <= lut[img_in[index]] <= 255 
}
