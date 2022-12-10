#!/bin/bash
mpicc -o contrast *.cpp
mpirun -n 1 ./contrast

# srun -N 1 -n 4 ./contrast
