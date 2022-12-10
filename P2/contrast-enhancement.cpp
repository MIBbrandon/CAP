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
    MPI_Bcast(&img_h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_w, 1, MPI_INT, 0, MPI_COMM_WORLD);

    result.h = img_h;
    result.w = img_w;

    // Ahora es necesario que rank==0 reparta img_in
    int * img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    int * img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    int max_num_elements_in_subset_img = repartidor(img_h * img_w, num_processors, img_counts, img_displs);  // Repartimos
    printf("R%d: w %d, h %d, np %d\n", rank, img_w, img_h, num_processors);

    // Ya sabemos como repartir
    // Cada proceso necesitará un array para coleccionar su subset de datos
    unsigned char * subset_img = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * subset_img_equalized = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));

    // SCATTERV para histograma
    // El rank==0 repartirá al resto de procesos lo que les corresponde
    printf("R%d starting scatter 1\n", rank);
    printf("senddispl0: %d, senddispl1: %d\n", img_displs[0], img_displs[1]);
    printf("sendcount0: %d, sendcount1: %d\n", img_counts[0], img_counts[1]);
    MPI_Scatterv(img_in.img, img_counts, img_displs, MPI_CHAR, subset_img, img_counts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);
    printf("R%d finished scatter 1\n", rank);

    histogram(hist, subset_img, img_counts[rank], 256);  // Cada proceso crea un histograma de su subset

    // Ahora bien, todos los procesos necesitan el histograma completo. Haremos que rank==0 los colleccione
    unsigned int * all_hists = (unsigned int *)malloc(256 * num_processors * sizeof(unsigned int));
    int * hist_counts = (int *)malloc(num_processors * sizeof(int));
    int * hist_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        hist_counts[i] = 256;
        hist_displs[i] = 256 * i;
    }

    MPI_Gatherv(hist, 256, MPI_INT, all_hists, hist_counts, hist_displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Suma todos los histogramas
    // Teniendo todos los histogramas en el mismo array, se pueden sumar elemento por elemento
    for(int i = 0; i < 256; i++) {
        hist[i] = 0;  // No se pierde el histograma de rank==0 porque está una copia en all_hists
        for(int j = 0; j < num_processors; j++) {
            hist[i] += all_hists[j * 256 + i];
        }
    }

    // Con rank==0 teniendo hist completo, lo debe compartir con el resto de procesos
    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&hist, 256, MPI_INT, 0, MPI_COMM_WORLD);

    // Ahora se puede hacer la equalización
    histogram_equalization_parallel(img_h * img_w, subset_img_equalized, subset_img, img_counts[rank], hist, 256);

    // Habiendo hecho todos los procesos su equalización correspondiente, queremos combinarlas todas en un solo array en el proceso de rank==0
    // Para ello, usaremos Gatherv
    if (rank == 0) {
        // Solo rank==0 necesitará alojar tanta memoria para recibir la imagen completa
        result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    }
    MPI_Gatherv(subset_img_equalized, img_counts[rank], MPI_CHAR, result.img, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);

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

PGM_IMG contrast_enhancement_g_OLD(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    int num_processors, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    // Repartimos el contar del histograma a todos los procesos por separado
    // Para ello, primero determinamos como repartir la imagen
    int * send_img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    // Ya que sabemos como van a ser distribuidos los elementos, usamos MPI_Scatterv para adjudicar los datos
    int * send_img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        send_img_displs[i] = 0;
    }
    // La función repartidor() rellena send_img_counts y send_img_displs con los valores adecuados para el Scatterv posterior
    int num_elements_in_subset_img = repartidor(result.w * result.h, num_processors, send_img_counts, send_img_displs);

    // Cada proceso necesitará un array para coleccionar su subset de datos
    unsigned char * subset_img = (unsigned char *)malloc(send_img_counts[rank] * sizeof(unsigned char));
    unsigned char * subset_img_equalized = (unsigned char *)malloc(send_img_counts[rank] * sizeof(unsigned char));

    // -------------------- SECCIÓN PARALELA ----------------------------
    // SCATTERV para histograma
    // MPI_Scatterv(img_in.img, send_img_counts, send_img_displs, MPI_CHAR, subset_img, send_img_counts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);

    // Si todos los procesos ya tienen img_in, no es necesario hacer Scatterv, solamente tienen que coger su trozo correspondiente de la imagen
    memcpy(subset_img, img_in.img + send_img_displs[rank], send_img_counts[rank] * sizeof(unsigned char));

    histogram(hist, subset_img, send_img_counts[rank], 256);  // Cada proceso crea un histograma de su subset

    // Ahora es necesario sumar elemento por elemento los valores en los histogramas
    // Para ello, recolectamos los histogramas en el root
    unsigned int * all_hists = (unsigned int *)malloc(256 * num_processors * sizeof(unsigned int));
    int * recv_hist_counts = (int *)malloc(num_processors * sizeof(int));
    int * recv_hist_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        recv_hist_counts[i] = 256;
        recv_hist_displs[i] = 256 * i;
    }
    
    // ALLGATHERV para histograma, así todos los procesos tienen una copia del histograma completo
    MPI_Allgatherv(hist, 256, MPI_INT, all_hists, recv_hist_counts, recv_hist_displs, MPI_INT, MPI_COMM_WORLD);
    

    // Todos los procesos tienen los histogramas de todos los demás
    // Teniendo todos los histogramas en el mismo array, se pueden sumar elemento por elemento
    for(int i = 0; i < 256; i++) {
        hist[i] = 0;
        for(int j = 0; j < num_processors; j++) {
            hist[i] += all_hists[j * 256 + i];
        }
    }

    // printf("R%d: %d, %d\n", rank, hist[0], hist[1]);
    
    // Al principio, cada proceso escribe su sección en su subset_img
    // Obtenemos el offset de cada proceso
    int corresponding_count = send_img_counts[rank];

    histogram_equalization_parallel(result.h * result.w, subset_img_equalized, subset_img, corresponding_count, hist, 256);
    
    // Sin embargo, queremos combinarlas todas en un solo array en el proceso de rank==0
    // Para ello, usaremos Gatherv
    MPI_Gatherv(subset_img_equalized, corresponding_count, MPI_CHAR, result.img, send_img_counts, send_img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    // -------------------- FIN DE SECCIÓN PARALELA ----------------------------

    free(send_img_counts);
    free(send_img_displs);
    free(subset_img);
    free(subset_img_equalized);
    free(recv_hist_counts);
    free(recv_hist_displs);
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

    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    yuv_med = rgb2yuv(img_in);
    y_equ = (unsigned char *)malloc(yuv_med.h*yuv_med.w*sizeof(unsigned char));

    // Repartimos el contar del histograma a todos los procesos por separado
    // Para ello, primero determinamos como repartir la imagen
    int * send_img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    // Ya que sabemos como van a ser distribuidos los elementos, usamos MPI_Scatterv para adjudicar los datos
    int * send_img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        send_img_displs[i] = 0;
    }
    // La función repartidor() rellena send_img_counts y send_img_displs con los valores adecuados para el Scatterv posterior
    int num_elements_in_subset_img = repartidor(yuv_med.w * yuv_med.h, num_processors, send_img_counts, send_img_displs);

    // Cada proceso necesitará un array para coleccionar su subset de datos
    unsigned char * subset_img = (unsigned char *)malloc(send_img_counts[rank] * sizeof(unsigned char));
    unsigned char * subset_img_equalized = (unsigned char *)malloc(send_img_counts[rank] * sizeof(unsigned char));

    // -------------------- SECCIÓN PARALELA ----------------------------
    // SCATTERV para histograma

    // Si todos los procesos ya tienen img_in, no es necesario hacer Scatterv, solamente tienen que coger su trozo correspondiente de la imagen
    memcpy(subset_img, yuv_med.img_y + send_img_displs[rank], send_img_counts[rank] * sizeof(unsigned char));
    
    histogram(hist, subset_img, send_img_counts[rank], 256);

    // Ahora es necesario sumar elemento por elemento los valores en los histogramas
    // Para ello, recolectamos los histogramas en el root
    unsigned int * all_hists = (unsigned int *)malloc(256 * num_processors * sizeof(unsigned int));
    int * recv_hist_counts = (int *)malloc(num_processors * sizeof(int));
    int * recv_hist_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        recv_hist_counts[i] = 256;
        recv_hist_displs[i] = 256 * i;
    }

    // ALLGATHERV para histograma, así todos los procesos tienen una copia del histograma completo
    MPI_Allgatherv(hist, 256, MPI_INT, all_hists, recv_hist_counts, recv_hist_displs, MPI_INT, MPI_COMM_WORLD);
    

    // Todos los procesos tienen los histogramas de todos los demás
    // Teniendo todos los histogramas en el mismo array, se pueden sumar elemento por elemento
    for(int i = 0; i < 256; i++) {
        hist[i] = 0;
        for(int j = 0; j < num_processors; j++) {
            hist[i] += all_hists[j * 256 + i];
        }
    }

    // printf("R%d: %d, %d\n", rank, hist[0], hist[1]);

    // Al principio, cada proceso escribe su sección en su subset_img
    // Obtenemos el offset de cada proceso
    int corresponding_count = send_img_counts[rank];

    histogram_equalization_parallel(yuv_med.h * yuv_med.w, subset_img_equalized, subset_img, corresponding_count, hist, 256);

    // Sin embargo, queremos combinarlas todas en un solo array en el proceso de rank==0
    // Para ello, usaremos Allgatherv (no solo Gatherv dado que el paso yuv2rgb supone que la imagen de entrada
    // yuv_med ya está accesible en todos los procesos, y Gatherv solo haría que el proceso de rank==0 tenga y_equ
    // completo)
    MPI_Allgatherv(subset_img_equalized, corresponding_count, MPI_CHAR, y_equ, send_img_counts, send_img_displs, MPI_CHAR, MPI_COMM_WORLD);

    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
    
    result = yuv2rgb(yuv_med);
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);
    
    free(send_img_counts);
    free(send_img_displs);
    free(subset_img);
    free(subset_img_equalized);
    free(recv_hist_counts);
    free(recv_hist_displs);
    free(all_hists);
    return result;
}

PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;
    
    unsigned char * l_equ;
    int hist[256];
    int num_processors, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hsl_med = rgb2hsl(img_in);
    l_equ = (unsigned char *)malloc(hsl_med.height * hsl_med.width*sizeof(unsigned char));

    // Repartimos el contar del histograma a todos los procesos por separado
    // Para ello, primero determinamos como repartir la imagen
    int * send_img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    // Ya que sabemos como van a ser distribuidos los elementos, usamos MPI_Scatterv para adjudicar los datos
    int * send_img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        send_img_displs[i] = 0;
    }
    // La función repartidor() rellena send_img_counts y send_img_displs con los valores adecuados para el Scatterv posterior
    int num_elements_in_subset_img = repartidor(hsl_med.width * hsl_med.height, num_processors, send_img_counts, send_img_displs);

    // Cada proceso necesitará un array para coleccionar su subset de datos
    unsigned char * subset_img = (unsigned char *)malloc(send_img_counts[rank] * sizeof(unsigned char));
    unsigned char * subset_img_equalized = (unsigned char *)malloc(send_img_counts[rank] * sizeof(unsigned char));

    // -------------------- SECCIÓN PARALELA ----------------------------
    // SCATTERV para histograma

    // Si todos los procesos ya tienen img_in, no es necesario hacer Scatterv, solamente tienen que coger su trozo correspondiente de la imagen
    memcpy(subset_img, hsl_med.l + send_img_displs[rank], send_img_counts[rank] * sizeof(unsigned char));

    histogram(hist, subset_img, send_img_counts[rank], 256);
    // histogram_equalization(l_equ, hsl_med.l,hist,hsl_med.width*hsl_med.height, 256);

    // Ahora es necesario sumar elemento por elemento los valores en los histogramas
    // Para ello, recolectamos los histogramas en el root
    unsigned int * all_hists = (unsigned int *)malloc(256 * num_processors * sizeof(unsigned int));
    int * recv_hist_counts = (int *)malloc(num_processors * sizeof(int));
    int * recv_hist_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        recv_hist_counts[i] = 256;
        recv_hist_displs[i] = 256 * i;
    }

    // ALLGATHERV para histograma, así todos los procesos tienen una copia del histograma completo
    MPI_Allgatherv(hist, 256, MPI_INT, all_hists, recv_hist_counts, recv_hist_displs, MPI_INT, MPI_COMM_WORLD);
    

    // Todos los procesos tienen los histogramas de todos los demás
    // Teniendo todos los histogramas en el mismo array, se pueden sumar elemento por elemento
    for(int i = 0; i < 256; i++) {
        hist[i] = 0;
        for(int j = 0; j < num_processors; j++) {
            hist[i] += all_hists[j * 256 + i];
        }
    }

    // printf("R%d: %d, %d\n", rank, hist[0], hist[1]);

    // Al principio, cada proceso escribe su sección en su subset_img
    // Obtenemos el offset de cada proceso
    int corresponding_count = send_img_counts[rank];

    histogram_equalization_parallel(hsl_med.width * hsl_med.height, subset_img_equalized, subset_img, corresponding_count, hist, 256);

    // Sin embargo, queremos combinarlas todas en un solo array en el proceso de rank==0
    // Para ello, usaremos Allgatherv (no solo Gatherv dado que el paso rgb2hsl supone que la imagen de entrada
    // hsl_med ya está accesible en todos los procesos, y Gatherv solo haría que el proceso de rank==0 tenga l_equ
    // completo)
    MPI_Allgatherv(subset_img_equalized, corresponding_count, MPI_CHAR, l_equ, send_img_counts, send_img_displs, MPI_CHAR, MPI_COMM_WORLD);
    
    free(hsl_med.l);
    hsl_med.l = l_equ;

    result = hsl2rgb(hsl_med);
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);

    free(send_img_counts);
    free(send_img_displs);
    free(subset_img);
    free(subset_img_equalized);
    free(recv_hist_counts);
    free(recv_hist_displs);
    free(all_hists);
    return result;
}


//Convert RGB to HSL, assume R,G,B in [0, 255]
//Output H, S in [0.0, 1.0] and L in [0, 255]
HSL_IMG rgb2hsl(PPM_IMG img_in)
{
    int i;
    float H, S, L;

    int num_processors, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    HSL_IMG img_out;// = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));

    // Cada pixel i requiere información de su correspondiente [r, g, b] y por lo tanto
    // los procesos se repartirán píxeles, no se repartirán tareas dentro de los píxeles.

    // Para ello, primero determinamos como repartir la imagen
    int * img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    // Ya que sabemos como van a ser distribuidos los elementos, usamos MPI_Scatterv para adjudicar los datos
    int * img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        img_displs[i] = 0;
    }
    // La función repartidor() rellena img_counts y img_displs con los valores adecuados para el Scatterv posterior
    int num_elements_in_subset_img = repartidor(img_in.w * img_in.h, num_processors, img_counts, img_displs);

    // Sabiendo la distribución, podemos preparar un buffer por cada canal para el Allgatherv posterior
    float * sendbuf_h = (float *)malloc(img_counts[rank] * sizeof(float));
    float * sendbuf_s = (float *)malloc(img_counts[rank] * sizeof(float));
    unsigned char * sendbuf_l = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));

    // printf("RGB2HSL %d: disp %d, count %d\n", rank, img_displs[rank], img_counts[rank]);
    
    // Cada proceso hace su parte localmente
    for(i = img_displs[rank]; i < img_displs[rank] + img_counts[rank]; i++){
        float var_r = ( (float)img_in.img_r[i]/255 );//Convert RGB to [0,1]
        float var_g = ( (float)img_in.img_g[i]/255 );
        float var_b = ( (float)img_in.img_b[i]/255 );
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

        sendbuf_h[i - img_displs[rank]] = H;
        sendbuf_s[i - img_displs[rank]] = S;
        sendbuf_l[i - img_displs[rank]] = (unsigned char)(L*255);
    }

    // Una vez todos los procesos hayan llenado sus buffers de sus secciones correspondientes,
    // coleccionamos los resultados canal por canal
    // Usamos Allgatherv para que cada proceso tenga la imagen completa, no solo el proceso de rank==0
    MPI_Allgatherv(sendbuf_h, img_counts[rank], MPI_FLOAT, img_out.h, img_counts, img_displs, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(sendbuf_s, img_counts[rank], MPI_FLOAT, img_out.s, img_counts, img_displs, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(sendbuf_l, img_counts[rank], MPI_CHAR, img_out.l, img_counts, img_displs, MPI_CHAR, MPI_COMM_WORLD);

    free(img_counts);
    free(img_displs);
    free(sendbuf_h);
    free(sendbuf_s);
    free(sendbuf_l);
    return img_out;
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
PPM_IMG hsl2rgb(HSL_IMG img_in)
{
    int i;
    PPM_IMG result;

    int num_processors, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    // Cada pixel i requiere información de su correspondiente [r, g, b] y por lo tanto
    // los procesos se repartirán píxeles, no se repartirán tareas dentro de los píxeles.

    // Para ello, primero determinamos como repartir la imagen
    int * img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    // Ya que sabemos como van a ser distribuidos los elementos, usamos MPI_Scatterv para adjudicar los datos
    int * img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        img_displs[i] = 0;
    }
    // La función repartidor() rellena img_counts y img_displs con los valores adecuados para el Scatterv posterior
    int num_elements_in_subset_img = repartidor(result.w * result.h, num_processors, img_counts, img_displs);

    // Sabiendo la distribución, podemos preparar un buffer por cada canal para el Allgatherv posterior
    unsigned char * sendbuf_r = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sendbuf_g = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sendbuf_b = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));

    // printf("HSL2RGB %d: disp %d, count %d\n", rank, img_displs[rank], img_counts[rank]);
    
    // Cada proceso hace su parte localmente
    for(i = img_displs[rank]; i < img_displs[rank] + img_counts[rank]; i++){
        float H = img_in.h[i];
        float S = img_in.s[i];
        float L = img_in.l[i]/255.0f;
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
        sendbuf_r[i - img_displs[rank]] = r;
        sendbuf_g[i - img_displs[rank]] = g;
        sendbuf_b[i - img_displs[rank]] = b;
    }

    // Una vez todos los procesos hayan llenado sus buffers de sus secciones correspondientes,
    // coleccionamos los resultados canal por canal
    
    // AVISO: este es el único paso donde hacemos Gatherv normal en vez de Allgatherv dado que este es el paso final, y solo el
    // proceso de rank==0 escribirá el fichero de salida. 
    MPI_Gatherv(sendbuf_r, img_counts[rank], MPI_CHAR, result.img_r, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(sendbuf_g, img_counts[rank], MPI_CHAR, result.img_g, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(sendbuf_b, img_counts[rank], MPI_CHAR, result.img_b, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);

    free(img_counts);
    free(img_displs);
    free(sendbuf_r);
    free(sendbuf_g);
    free(sendbuf_b);
    return result;
}

//Convert RGB to YUV, all components in [0, 255]
YUV_IMG rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
    int i;//, j;
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    int num_processors, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    // Repartimos los procesos a lo largo de la imagen, puesto que para cada canal en [y, u, v] es necesario
    // tener información sobre todos los canales en [r, g, b]

    // Para ello, primero determinamos como repartir la imagen
    int * img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    // Ya que sabemos como van a ser distribuidos los elementos, usamos MPI_Scatterv para adjudicar los datos
    int * img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        img_displs[i] = 0;
    }
    // La función repartidor() rellena img_counts y img_displs con los valores adecuados para el Scatterv posterior
    int num_elements_in_subset_img = repartidor(img_out.w * img_out.h, num_processors, img_counts, img_displs);

    // Sabiendo la distribución, podemos preparar un buffer por cada canal para el Allgatherv posterior
    unsigned char * sendbuf_y = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sendbuf_u = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sendbuf_v = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));

    // printf("RGB2YUV %d: disp %d, count %d\n", rank, img_displs[rank], img_counts[rank]);

    // Cada proceso hace su parte localmente
    for(i = img_displs[rank]; i < img_displs[rank] + img_counts[rank]; i++){
        // Todos los procesos tienen img_in ya en memoria con la lectura
        r = img_in.img_r[i];
        g = img_in.img_g[i];
        b = img_in.img_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        sendbuf_y[i - img_displs[rank]] = y;
        sendbuf_u[i - img_displs[rank]] = cb;
        sendbuf_v[i - img_displs[rank]] = cr;
    }

    // Una vez todos los procesos hayan llenado sus buffers de sus secciones correspondientes,
    // coleccionamos los resultados canal por canal
    // Usamos Allgatherv para que cada proceso tenga la imagen completa, no solo el proceso de rank==0
    MPI_Allgatherv(sendbuf_y, img_counts[rank], MPI_CHAR, img_out.img_y, img_counts, img_displs, MPI_CHAR, MPI_COMM_WORLD);
    MPI_Allgatherv(sendbuf_u, img_counts[rank], MPI_CHAR, img_out.img_u, img_counts, img_displs, MPI_CHAR, MPI_COMM_WORLD);
    MPI_Allgatherv(sendbuf_v, img_counts[rank], MPI_CHAR, img_out.img_v, img_counts, img_displs, MPI_CHAR, MPI_COMM_WORLD);
    
    free(img_counts);
    free(img_displs);
    free(sendbuf_y);
    free(sendbuf_u);
    free(sendbuf_v);
    return img_out;
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
PPM_IMG yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    int i;
    int rt, gt, bt;
    int y, cb, cr;
    int num_processors, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    // Repartimos los procesos a lo largo de la imagen, puesto que para cada canal en [r, g, b] es necesario
    // tener información sobre todos los canales en [y, u, v]

    // Para ello, primero determinamos como repartir la imagen
    int * img_counts = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI cuántos elementos enivará a cada proceso
    // Ya que sabemos como van a ser distribuidos los elementos, usamos MPI_Scatterv para adjudicar los datos
    int * img_displs = (int *)malloc(num_processors * sizeof(int));  // Le dice a MPI donde en el array empezar a poner de cada proceso
    for(int i = 0; i < num_processors; i++) {
        img_displs[i] = 0;
    }
    // La función repartidor() rellena img_counts y img_displs con los valores adecuados para el Scatterv posterior
    int num_elements_in_subset_img = repartidor(img_out.w * img_out.h, num_processors, img_counts, img_displs);

    // Sabiendo la distribución, podemos preparar un buffer por cada canal para el Allgatherv posterior
    unsigned char * sendbuf_r = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sendbuf_g = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));
    unsigned char * sendbuf_b = (unsigned char *)malloc(img_counts[rank] * sizeof(unsigned char));

    // printf("YUV2RGB %d: disp %d, count %d\n", rank, img_displs[rank], img_counts[rank]);
    
    for(i = img_displs[rank]; i < img_displs[rank] + img_counts[rank]; i++){
        // Todos los procesos tienen img_in ya en memoria con la lectura
        y  = (int)img_in.img_y[i];
        cb = (int)img_in.img_u[i] - 128;
        cr = (int)img_in.img_v[i] - 128;

        rt = (int)( y + 1.402*cr);
        gt = (int)( y - 0.344*cb - 0.714*cr);
        bt = (int)( y + 1.772*cb);

        sendbuf_r[i - img_displs[rank]] = clip_rgb(rt);
        sendbuf_g[i - img_displs[rank]] = clip_rgb(gt);
        sendbuf_b[i - img_displs[rank]] = clip_rgb(bt);

    }
    

    // Una vez todos los procesos hayan llenado sus buffers de sus secciones correspondientes,
    // coleccionamos los resultados canal por canal

    // AVISO: este es el único paso donde hacemos Gatherv normal en vez de Allgatherv dado que este es el paso final, y solo el
    // proceso de rank==0 escribirá el fichero de salida. 
    MPI_Gatherv(sendbuf_r, img_counts[rank], MPI_CHAR, img_out.img_r, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(sendbuf_g, img_counts[rank], MPI_CHAR, img_out.img_g, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(sendbuf_b, img_counts[rank], MPI_CHAR, img_out.img_b, img_counts, img_displs, MPI_CHAR, 0, MPI_COMM_WORLD);

    free(img_counts);
    free(img_displs);
    free(sendbuf_r);
    free(sendbuf_g);
    free(sendbuf_b);
    return img_out;
}
