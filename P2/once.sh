#!/bin/bash

# Compilando con make y ejecutando con srun (o mpirun), obtenemos la mejor combinaciónde tiempos grey time y HSL time
make
salloc -N2 srun -n6 ./contrast

# Compilando con mpicc y ejecutando con srun (o mpirun), solo mejora la grey time, no la HSL time
# mpicc -o contrast_mpicc *.cpp
# mpirun -n 1 ./contrast_mpicc
