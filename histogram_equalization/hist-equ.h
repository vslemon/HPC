#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

/*
void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in,int * hist_in, int img_size, int nbr_bin);
*/

__global__ void histogram_g1(int * hist_out , unsigned char * img_in, int size_of_img);
__global__ void histogram_g2(int * hist_out, unsigned char * img_in,int *block_histogram,int size_of_img);
__global__ void histogram_s1(int * hist_out, unsigned char * img_in, int size_of_img);
__global__ void histogram_s3(int * hist_out , unsigned char * img_in , int size_of_img);
__global__ void histogram_s4(int * hist_out , unsigned char * img_in ,int * block_histogram, int size_of_img);
__global__ void result_calc(unsigned char * img_out,int * lut,unsigned int img_size);
__global__ void result_calc_param(unsigned char * img_out, int * lut);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);

#endif
