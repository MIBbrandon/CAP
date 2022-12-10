#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
using namespace std;

// El único objetivo de este es proporcionar funciones de utilidad

int repartidor(int total_num_elements, int num_processors, int *send_counts, int *displs, int forced_chunk_size = -1) {
    // Poner en send_counts el número de elementos que va a recibir cada proceso.
    // Por defecto, reparte equitativamente, pero si se especifica forced_chunk_size, serán repartidas
    // con ese número de elementos, siendo el resto dejado para el último proceso

    
    if (forced_chunk_size > 0) {
        // Si se especifica forced_chunk_size
        int adjudicado = 0;
        int i = 0;
        while(adjudicado < total_num_elements) {
            send_counts[i] = std::min(forced_chunk_size, total_num_elements - adjudicado);
            adjudicado += forced_chunk_size;
            i++;
        }
        // El resto de elementos que quedan son adjudicados al último proceso
        send_counts[num_processors - 1] += std::max(0, total_num_elements - (num_processors * forced_chunk_size));

        // Es necesario devolver el máximo tamaño adjudicado a cualquier elemento. O será el valor del primero (
        // el mismo valor que de todos excepto el último) o será el último
        return std::max(send_counts[0], send_counts[num_processors - 1]);

    } else {
        // Por defecto, se reparte equitativamente
        int ideal_chunk_size = total_num_elements / num_processors;  // Integer division
        int num_elements_remaining = total_num_elements % num_processors;  // Remainder

        int offset = 0;
        for(int i = 0; i < num_processors; i++) {
            send_counts[i] = ideal_chunk_size;
            displs[i] = offset;  // Displacement within the array

            // Ya que accedemos en memoria, le dejamos el sobrante que le tocará
            if (num_elements_remaining > 0) {
                send_counts[i]++;
                num_elements_remaining--;
            }
            
            offset += send_counts[i];  // Update where to write next
        }

        // Si se reparte de esta manera, es garantizado que el primero representará el máximo del conjunto
        return send_counts[0];
    }
}

// int main() {
//     int MAX_PROCESSES = 5;
//     int total_num_elements = 11;
//     int num_processors = 3;
//     int send_counts[MAX_PROCESSES];

//     int devuelto = repartidor(total_num_elements, num_processors, send_counts);

//     for (int i = 0; i < MAX_PROCESSES; i++) {
//         printf("%d ", send_counts[i]);
//     }

//     printf("Devuelto: %d\n", devuelto);

//     return 0;
// }

