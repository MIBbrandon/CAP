#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    int w_size, w_rank, name_len;
    char p_name[MPI_MAX_PROCESSOR_NAME];
    int send_vec[9], recv_vec[9], i;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);
    printf("Size: %d | Rank: %d\n", w_size, w_rank);
    MPI_Get_processor_name(p_name, &name_len);
    if (w_rank == 0) {
        for (i=0; i<9; i++) send_vec[i]=i;
    }
    MPI_Scatter(send_vec, 9/w_size, MPI_INT, recv_vec, 9, MPI_INT, 0, MPI_COMM_WORLD);
    for (i=0; i < 9/w_size; i++) {
        printf("(%s): Proc * num: %d * %d = %d\n", p_name, w_rank, recv_vec[i], w_rank*recv_vec[i]);
    }
    MPI_Finalize();
}