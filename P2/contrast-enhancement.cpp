#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <mpi.h>

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    // Suponemos que solo rank==0 tiene img_in bien
    // Todos los procesos tienen img_h y img_w
    
    PGM_IMG result;
    int hist[256];
    int num_processors, rank;
    int img_h, img_w;  // To be received by rank==0

    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Primero, rank==0 tiene que coger las dimensiones
        img_h = img_in.h;
        img_w = img_in.w;
    }

    // From rank==0, broadcast dimensions of the image
    double g_broadcast_image_dimensions_start = MPI_Wtime();
    MPI_Bcast(&img_h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double g_broadcast_image_dimensions_end = MPI_Wtime();

    result.h = img_h;
    result.w = img_w;

    // Ahora es necesario que rank==0 reparta img_in
    int * img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    int * img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    int max_num_elements_in_subset_img = repartidor(img_h * img_w, num_processors, img_counts, img_displs);  // Repartimos

    // Ya sabemos como repartir
    // Cada proceso necesitará un array para coleccionar su subset de datos
    unsigned char * subset_img = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * subset_img_equalized = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));

    // SCATTERV para histograma
    // El rank==0 repartirá al resto de procesos lo que les corresponde
    double g_scatter_image_start = MPI_Wtime();
    MPI_Scatterv(img_in.img, img_counts, img_displs, MPI_CHAR, subset_img, img_counts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);
    double g_scatter_image_end = MPI_Wtime();


    double g_partial_histogram_parallel_start = MPI_Wtime();
    histogram(hist, subset_img, img_counts[rank], 256);  // Cada proceso crea un histograma de su subset
    double g_partial_histogram_parallel_end = MPI_Wtime();

    // Ahora bien, todos los procesos necesitan el histograma completo. Haremos que rank==0 los colleccione
    unsigned int * all_hists = (unsigned int *)malloc(256 * num_processors * sizeof(unsigned int));
    int * hist_counts = (int *)malloc(num_processors * sizeof(int));
    int * hist_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        hist_counts[i] = 256;
        hist_displs[i] = 256 * i;
    }

    double g_gather_hists_start = MPI_Wtime();
    MPI_Gatherv(hist, 256, MPI_INT, all_hists, hist_counts, hist_displs, MPI_INT, 0, MPI_COMM_WORLD);
    double g_gather_hists_end = MPI_Wtime();

    double g_add_hists_r0_start = MPI_Wtime();
    // Solo rank==0 hace esta suma
    if (rank == 0) {
        for(int i = 0; i < 256; i++) {
            hist[i] = 0;  // No se pierde el histograma de rank==0 porque está una copia en all_hists
            for(int j = 0; j < num_processors; j++) {
                hist[i] += all_hists[j * 256 + i];
            }
        }
    }
    double g_add_hists_r0_end = MPI_Wtime();
    

    // Con rank==0 teniendo hist completo, lo debe compartir con el resto de procesos
    // MPI_Barrier(MPI_COMM_WORLD);
    double g_broadcast_done_hist_start = MPI_Wtime();
    MPI_Bcast(&hist, 256, MPI_INT, 0, MPI_COMM_WORLD);
    double g_broadcast_done_hist_end = MPI_Wtime();

    // Ahora se puede hacer la equalización
    double g_histogram_equ_parallel_start = MPI_Wtime();
    histogram_equalization_parallel(img_h * img_w, subset_img_equalized, subset_img, img_counts[rank], hist, 256);
    double g_histogram_equ_parallel_end = MPI_Wtime();

    // Habiendo hecho todos los procesos su equalización correspondiente, queremos combinarlas todas en un solo array en el proceso de rank==0
    // Para ello, usaremos Gatherv
    if (rank == 0) {
        // Solo rank==0 necesitará alojar tanta memoria para recibir la imagen completa
        result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    }

    double g_gather_full_result_start = MPI_Wtime();
    MPI_Gatherv(subset_img_equalized, img_counts[rank], MPI_CHAR, result.img, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    double g_gather_full_result_end = MPI_Wtime();

    if (rank == 0) {
        printf("R%d grey: "
                "\n\tbroadcast_image_dimensions: %f"
                "\n\tscatter_image: %f"
                "\n\tpartial_histogram_parallel: %f"
                "\n\tgather_hists: %f"
                "\n\tadd_hists_r0: %f"
                "\n\tbroadcast_done_hist: %f"
                "\n\thistogram_equ_parallel: %f"
                "\n\tgather_full_result: %f"
                "\n",
                rank,
                (g_broadcast_image_dimensions_end - g_broadcast_image_dimensions_start),
                (g_scatter_image_end - g_scatter_image_start),
                (g_partial_histogram_parallel_end - g_partial_histogram_parallel_start),
                (g_gather_hists_end - g_gather_hists_start),
                (g_add_hists_r0_end - g_add_hists_r0_start),
                (g_broadcast_done_hist_end - g_broadcast_done_hist_start),
                (g_histogram_equ_parallel_end - g_histogram_equ_parallel_start),
                (g_gather_full_result_end - g_gather_full_result_start)
                );
    }

    // rank==0 ya tiene todos los pixeles en result.img, solo necesita devolver result
    free(img_counts);
    free(img_displs);
    free(subset_img);
    free(subset_img_equalized);
    free(hist_counts);
    free(hist_displs);
    free(all_hists);
    return result;
}

PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;
    
    unsigned char * y_equ;
    int hist[256];
    int num_processors, rank;
    int img_h, img_w;  // To be received by rank==0

    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Primero, rank==0 tiene que coger las dimensiones
        result.w = img_in.w;
        result.h = img_in.h;
        img_h = img_in.h;
        img_w = img_in.w;
    }

    // From rank==0, broadcast dimensions of the image
    MPI_Bcast(&img_h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // yuv_med = rgb2yuv(img_in);
    // y_equ = (unsigned char *)malloc(yuv_med.h*yuv_med.w*sizeof(unsigned char));

    // Primero determinamos como repartir la imagen
    int * img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    // Ya que sabemos como van a ser distribuidos los elementos, usamos MPI_Scatterv para adjudicar los datos
    int * img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        img_displs[i] = 0;
    }
    // La función repartidor() rellena img_counts y img_displs con los valores adecuados para el Scatterv posterior
    int num_elements_in_subset_img = repartidor(img_h * img_w, num_processors, img_counts, img_displs);

    // Conociendo la distribución, rank==0 debe repartir
    // Para repartir, todos los procesos tienen que tener la memoria adecuada preparada
    // Vectores que guardan valores RGB
    unsigned char * sub_img_r_vector = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sub_img_g_vector = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sub_img_b_vector = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    // Vectores que guardan valores YUV
    unsigned char * sub_img_y_vector = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sub_img_u_vector = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sub_img_v_vector = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    // Se hará equalización del canal Y, y su resultado se guardará aquí
    unsigned char * sub_img_y_vector_equ = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));

    // El rank==0 repartirá al resto de procesos lo que les corresponde
    // Solo hay que hacer scatter de los canales RGB
    MPI_Scatterv(img_in.img_r, img_counts, img_displs, MPI_CHAR, sub_img_r_vector, img_counts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_g, img_counts, img_displs, MPI_CHAR, sub_img_g_vector, img_counts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_b, img_counts, img_displs, MPI_CHAR, sub_img_b_vector, img_counts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);

    // Cada proceso transforma su sección de la imagen de RGB a YUV
    rgb2yuv(sub_img_r_vector, sub_img_g_vector, sub_img_b_vector, 
            sub_img_y_vector, sub_img_u_vector, sub_img_v_vector, 
            img_counts[rank], img_h, img_w);
    
    // Es necesario construir un histograma del canal Y
    // Cada proceso hace su histograma individual
    histogram(hist, sub_img_y_vector, img_counts[rank], 256);

    // Ahora es necesario sumar elemento por elemento los valores en los histogramas
    // Para ello, recolectamos los histogramas en el root
    unsigned int * all_hists = (unsigned int *)malloc(256 * num_processors * sizeof(unsigned int));
    int * hist_counts = (int *)malloc(num_processors * sizeof(int));
    int * hist_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        hist_counts[i] = 256;
        hist_displs[i] = 256 * i;
    }

    // Proceso rank==0 coleccionará los histogramas individuales para hacer el histograma completo
    MPI_Gatherv(hist, 256, MPI_INT, all_hists, hist_counts, hist_displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Solo rank==0 hace esta suma
    if (rank == 0) {
        #pragma omp for
        for(int i = 0; i < 256; i++) {
            hist[i] = 0;  // No se pierde el histograma de rank==0 porque está una copia en all_hists
            for(int j = 0; j < num_processors; j++) {
                hist[i] += all_hists[j * 256 + i];
            }
        }
    }

    // Con rank==0 teniendo hist completo, lo debe compartir con el resto de procesos
    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&hist, 256, MPI_INT, 0, MPI_COMM_WORLD);

    // printf("R%d: %d, %d\n", rank, hist[0], hist[1]);

    // Ahora se puede hacer la equalización (de nuevo, cada proceso hace su parte individualmente)
    histogram_equalization_parallel(img_h * img_w, sub_img_y_vector_equ, sub_img_y_vector, img_counts[rank], hist, 256);

    // Convertimos de nuevo a RGB, pero usando YUV con el canal Y equalizado
    yuv2rgb(sub_img_y_vector_equ, sub_img_u_vector, sub_img_v_vector,
                sub_img_r_vector, sub_img_g_vector, sub_img_b_vector, 
                img_counts[rank], img_h, img_w);

    if (rank == 0) {
        // Solo rank==0 necesitará alojar tanta memoria para recibir la imagen completa
        result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    }

    // Ya se han cumplido todos los pasos, rank==0 debe recoger todas las partes para producir el resultado completo deseado
    MPI_Gatherv(sub_img_r_vector, img_counts[rank], MPI_CHAR, result.img_r, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(sub_img_g_vector, img_counts[rank], MPI_CHAR, result.img_g, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(sub_img_b_vector, img_counts[rank], MPI_CHAR, result.img_b, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);

    free(img_counts);
    free(img_displs);

    free(sub_img_r_vector);
    free(sub_img_g_vector);
    free(sub_img_b_vector);
    free(sub_img_y_vector);
    free(sub_img_y_vector_equ);
    free(sub_img_u_vector);
    free(sub_img_v_vector);

    free(hist_counts);
    free(hist_displs);
    free(all_hists);
    return result;
}

PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in)
{
    PPM_IMG result;
    
    int hist[256];
    int num_processors, rank;
    int img_h, img_w;  // To be received by rank==0

    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Primero, rank==0 tiene que coger las dimensiones
        result.w = img_in.w;
        result.h = img_in.h;
        img_h = img_in.h;
        img_w = img_in.w;
    }

    // From rank==0, broadcast dimensions of the image
    MPI_Bcast(&img_h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_w, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // Primero determinamos como repartir la imagen
    int * img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    // Ya que sabemos como van a ser distribuidos los elementos, usamos MPI_Scatterv para adjudicar los datos
    int * img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        img_displs[i] = 0;
    }
    // La función repartidor() rellena img_counts y img_displs con los valores adecuados para el Scatterv posterior
    int num_elements_in_subset_img = repartidor(img_h * img_w, num_processors, img_counts, img_displs);


    // Conociendo la distribución, rank==0 debe repartir
    // Para repartir, todos los procesos tienen que tener la memoria adecuada preparada
    // Vectores que guardan valores RGB
    unsigned char * sub_img_r_vector = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sub_img_g_vector = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sub_img_b_vector = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    // Vectores que guardan valores HSL
    float * sub_img_h_vector = (float *)malloc(img_counts[rank] * sizeof(float));
    float * sub_img_s_vector = (float *)malloc(img_counts[rank] * sizeof(float));
    unsigned char * sub_img_l_vector = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    // Se hará equalización del canal L, y su resultado se guardará aquí
    unsigned char * sub_img_l_vector_equ = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));


    // El rank==0 repartirá al resto de procesos lo que les corresponde
    // Solo hay que hacer scatter de los canales RGB
    MPI_Scatterv(img_in.img_r, img_counts, img_displs, MPI_CHAR, sub_img_r_vector, img_counts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_g, img_counts, img_displs, MPI_CHAR, sub_img_g_vector, img_counts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_b, img_counts, img_displs, MPI_CHAR, sub_img_b_vector, img_counts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);

    // Cada proceso transforma su sección de la imagen de RGB a HSL
    rgb2hsl(sub_img_r_vector, sub_img_g_vector, sub_img_b_vector, 
            sub_img_h_vector, sub_img_s_vector, sub_img_l_vector, 
            img_counts[rank], img_h, img_w);
    
    // Es necesario construir un histograma del canal L
    // Cada proceso hace su histograma individual
    histogram(hist, sub_img_l_vector, img_counts[rank], 256);

    // Ahora es necesario sumar elemento por elemento los valores en los histogramas
    // Para ello, recolectamos los histogramas en el root
    unsigned int * all_hists = (unsigned int *)malloc(256 * num_processors * sizeof(unsigned int));
    int * hist_counts = (int *)malloc(num_processors * sizeof(int));
    int * hist_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        hist_counts[i] = 256;
        hist_displs[i] = 256 * i;
    }

    // Proceso rank==0 coleccionará los histogramas individuales para hacer el histograma completo
    MPI_Gatherv(hist, 256, MPI_INT, all_hists, hist_counts, hist_displs, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Solo rank==0 hace esta suma
    if (rank == 0) {
        #pragma omp for
        for(int i = 0; i < 256; i++) {
            hist[i] = 0;  // No se pierde el histograma de rank==0 porque está una copia en all_hists
            for(int j = 0; j < num_processors; j++) {
                hist[i] += all_hists[j * 256 + i];
            }
        }
    }

    // Con rank==0 teniendo hist completo, lo debe compartir con el resto de procesos
    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&hist, 256, MPI_INT, 0, MPI_COMM_WORLD);

    // printf("R%d: %d, %d\n", rank, hist[0], hist[1]);

    // Ahora se puede hacer la equalización (de nuevo, cada proceso hace su parte individualmente)
    histogram_equalization_parallel(img_h * img_w, sub_img_l_vector_equ, sub_img_l_vector, img_counts[rank], hist, 256);
    
    // Convertimos de nuevo a RGB, pero usando HSL con el canal L equalizado
    hsl2rgb(sub_img_h_vector, sub_img_s_vector, sub_img_l_vector_equ,
                sub_img_r_vector, sub_img_g_vector, sub_img_b_vector, 
                img_counts[rank], img_h, img_w);

    if (rank == 0) {
        // Solo rank==0 necesitará alojar tanta memoria para recibir la imagen completa
        result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    }

    // Ya se han cumplido todos los pasos, rank==0 debe recoger todas las partes para producir el resultado completo deseado
    MPI_Gatherv(sub_img_r_vector, img_counts[rank], MPI_CHAR, result.img_r, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(sub_img_g_vector, img_counts[rank], MPI_CHAR, result.img_g, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(sub_img_b_vector, img_counts[rank], MPI_CHAR, result.img_b, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);

    free(img_counts);
    free(img_displs);

    free(sub_img_r_vector);
    free(sub_img_g_vector);
    free(sub_img_b_vector);
    free(sub_img_h_vector);
    free(sub_img_s_vector);
    free(sub_img_l_vector);
    free(sub_img_l_vector_equ);

    free(hist_counts);
    free(hist_displs);
    free(all_hists);
    return result;
}


//Convert RGB to HSL, assume R,G,B in [0, 255]
//Output H, S in [0.0, 1.0] and L in [0, 255]
void rgb2hsl(unsigned char * sub_img_r_vector, unsigned char * sub_img_g_vector, unsigned char * sub_img_b_vector, 
                float * sub_img_h_vector, float * sub_img_s_vector, unsigned char * sub_img_l_vector, 
                int num_assigned_pixels, int img_h, int img_w)
{
    // Every process here is completely independent doing the same thing on different data (SIMD)
    int i;
    float H, S, L;

    // Cada pixel i requiere información de su correspondiente [r, g, b] y por lo tanto
    // los procesos se repartirán píxeles, no se repartirán tareas dentro de los píxeles.

    // printf("RGB2HSL %d: disp %d, count %d\n", rank, img_displs[rank], img_counts[rank]);
    
    // Cada proceso hace su parte localmente
    #pragma omp for
    for(i = 0; i < num_assigned_pixels; i++){
        float var_r = ( (float)sub_img_r_vector[i]/255 );//Convert RGB to [0,1]
        float var_g = ( (float)sub_img_g_vector[i]/255 );
        float var_b = ( (float)sub_img_b_vector[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 )//This is a gray, no chroma...
        {
            H = 0;         
            S = 0;    
        }
        else                                    //Chromatic data...
        {
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;
            }
            else{       
                if( var_g == var_max ){
                    H = (1.0/3.0) + del_r - del_b;
                }
                else{
                        H = (2.0/3.0) + del_g - del_r;
                }   
            }
            
        }
        
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

        sub_img_h_vector[i] = H;
        sub_img_s_vector[i] = S;
        sub_img_l_vector[i] = (unsigned char)(L*255);
    }


}

float Hue_2_RGB( float v1, float v2, float vH )             //Function Hue_2_RGB
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}

//Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
//Output R,G,B in [0, 255]
// void hsl2rgb(HSL_IMG img_in)
void hsl2rgb(float * sub_img_h_vector, float * sub_img_s_vector, unsigned char * sub_img_l_vector,
                unsigned char * sub_img_r_vector, unsigned char * sub_img_g_vector, unsigned char * sub_img_b_vector, 
                int num_assigned_pixels, int img_h, int img_w)
{
    // Every process here is completely independent doing the same thing on different data (SIMD)
    int i;

    // Cada pixel i requiere información de su correspondiente [r, g, b] y por lo tanto
    // los procesos se repartirán píxeles, no se repartirán tareas dentro de los píxeles.

    // printf("HSL2RGB %d: disp %d, count %d\n", rank, img_displs[rank], img_counts[rank]);
    
    // Cada proceso hace su parte localmente
    #pragma omp parallel for  // El que más impacto tiene para bien. El único que usando parallel for funciona mejor que solo for
    for(i = 0; i < num_assigned_pixels; i++){
        float H = sub_img_h_vector[i];
        float S = sub_img_s_vector[i];
        float L = sub_img_l_vector[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r, g, b;
        
        if ( S == 0 )
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {
            
            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            var_1 = 2 * L - var_2;
            r = 255 * Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
            g = 255 * Hue_2_RGB( var_1, var_2, H );
            b = 255 * Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
        }
        sub_img_r_vector[i] = r;
        sub_img_g_vector[i] = g;
        sub_img_b_vector[i] = b;
    }
}

//Convert RGB to YUV, all components in [0, 255]
void rgb2yuv(unsigned char * sub_img_r_vector, unsigned char * sub_img_g_vector, unsigned char * sub_img_b_vector, 
                unsigned char * sub_img_y_vector, unsigned char * sub_img_u_vector, unsigned char * sub_img_v_vector, 
                int num_assigned_pixels, int img_h, int img_w)
{
    // Every process here is completely independent doing the same thing on different data (SIMD)
    int i;//, j;
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    
    // Cada pixel i requiere información de su correspondiente [r, g, b] y por lo tanto
    // los procesos se repartirán píxeles, no se repartirán tareas dentro de los píxeles.
    
    // printf("RGB2YUV %d: disp %d, count %d\n", rank, img_displs[rank], img_counts[rank]);

    // Cada proceso hace su parte localmente
    #pragma omp for // El que más impacto tiene para mal si usamos parallel
    for(i = 0; i < num_assigned_pixels; i++){
        // Todos los procesos tienen img_in ya en memoria con la lectura
        r = sub_img_r_vector[i];
        g = sub_img_g_vector[i];
        b = sub_img_b_vector[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        sub_img_y_vector[i] = y;
        sub_img_u_vector[i] = cb;
        sub_img_v_vector[i] = cr;
    }
}

unsigned char clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

//Convert YUV to RGB, all components in [0, 255]
void yuv2rgb(unsigned char * sub_img_y_vector, unsigned char * sub_img_u_vector, unsigned char * sub_img_v_vector,
                unsigned char * sub_img_r_vector, unsigned char * sub_img_g_vector, unsigned char * sub_img_b_vector, 
                int num_assigned_pixels, int img_h, int img_w)
{
    // Every process here is completely independent doing the same thing on different data (SIMD)
    int i;
    int rt, gt, bt;
    int y, cb, cr;

    // Cada pixel i requiere información de su correspondiente [r, g, b] y por lo tanto
    // los procesos se repartirán píxeles, no se repartirán tareas dentro de los píxeles.

    // printf("YUV2RGB %d: disp %d, count %d\n", rank, img_displs[rank], img_counts[rank]);
    
    // Cada proceso hace su parte localmente
    #pragma omp for // El que más impacto tiene para mal si usamos parallel
    for(i = 0; i < num_assigned_pixels; i++){
        y  = (int)sub_img_y_vector[i];
        cb = (int)sub_img_u_vector[i] - 128;
        cr = (int)sub_img_v_vector[i] - 128;

        rt = (int)( y + 1.402*cr);
        gt = (int)( y - 0.344*cb - 0.714*cr);
        bt = (int)( y + 1.772*cb);

        sub_img_r_vector[i] = clip_rgb(rt);
        sub_img_g_vector[i] = clip_rgb(gt);
        sub_img_b_vector[i] = clip_rgb(bt);
    }
}
