#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <mpi.h>
#include <algorithm>

using namespace std;

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    #pragma omp for
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    #pragma omp for
    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization_parallel(int img_out_size, unsigned char * subset_img_equalized, unsigned char * subset_img_in, int subset_img_in_size, int * hist_in, int nbr_bin){
    // Not worth parallelizing this first part, only 256 elements
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d, temp_lut_i;

    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_out_size - min;
    #pragma omp for
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        temp_lut_i = (int)(((float)cdf - min)*255/d + 0.5);


        // Los valores que entran y salen de lut son limitados al rango de 0 a 255. Por tanto, para evitar
        // varios ifs en la imagen completa, limitamos los lut ya aquí
        lut[i] = std::min(std::max(temp_lut_i, 0), 255);
    }
    
    // However, this goes over the entire image, so it is worth parallelizing
    /* Get the result image */
    // Each process does its own section, so they write the OUTPUT (not the input) at different places depending on their rank
    #pragma omp for
    for(i = 0; i < subset_img_in_size; i++){
        subset_img_equalized[i] = (unsigned char)lut[subset_img_in[i]];
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    #pragma omp for
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
        
    }
    
    /* Get the result image */
    #pragma omp for
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}


