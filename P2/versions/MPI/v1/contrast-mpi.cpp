#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "hist-equ.h"
#include <utility>
#include <fstream>

double run_cpu_gray_test(PGM_IMG img_in);
std::pair<double, double> run_cpu_color_test(PPM_IMG img_in);

using namespace std;

int main(int argc, char *argv[]){
    MPI_Init( &argc, &argv );

    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("VERSION 1\n");
        printf("Running contrast enhancement for gray-scale images.\n");
    }
    
    img_ibuf_g = read_pgm("in.pgm");
    double gray_time = run_cpu_gray_test(img_ibuf_g);
    free_pgm(img_ibuf_g);
    
    if (rank == 0) {
        printf("Running contrast enhancement for color images.\n");
    }
    
    img_ibuf_c = read_ppm("in.ppm");
    std::pair<double, double> colours_times = run_cpu_color_test(img_ibuf_c);
    double hsl_time = colours_times.first;
    double yuv_time = colours_times.second;
    free_ppm(img_ibuf_c);
    
    MPI_Finalize();

    return 0;
}

double run_cpu_gray_test(PGM_IMG img_in)
{
    PGM_IMG img_obuf;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Starting CPU processing...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start_grey = MPI_Wtime();
    img_obuf = contrast_enhancement_g(img_in);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_grey = MPI_Wtime();
    double duration_grey = end_grey - start_grey;
    
    printf("Grey %d processing time: %f (s)\n", rank, duration_grey /* TIMER */ );

    if (rank == 0) {
        write_pgm(img_obuf, "out.pgm");
    }
    
    free_pgm(img_obuf);
    return duration_grey;
}

std::pair<double, double> run_cpu_color_test(PPM_IMG img_in)
{
    PPM_IMG img_obuf_hsl, img_obuf_yuv;

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        printf("Starting CPU processing...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_colour_hsl = MPI_Wtime();
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_colour_hsl = MPI_Wtime();
    double duration_colour_hsl = end_colour_hsl - start_colour_hsl;
    printf("HSL %d processing time: %f (s)\n", rank, duration_colour_hsl /* TIMER */ );
    
    if (rank == 0) {
        write_ppm(img_obuf_hsl, "out_hsl.ppm");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_colour_yuv = MPI_Wtime();
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_colour_yuv = MPI_Wtime();
    double duration_colour_yuv = end_colour_yuv - start_colour_yuv;
    printf("YUV %d processing time: %f (s)\n", rank, duration_colour_yuv /* TIMER */);
    
    if (rank == 0) {
        write_ppm(img_obuf_yuv, "out_yuv.ppm");
    }
    
    
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);

    return std::make_pair(duration_colour_hsl, duration_colour_yuv);
}



PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    // printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    // printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}

