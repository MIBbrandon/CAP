Práctica 2 - Computación de Altas Prestaciones
=======


## Requisitos

Dado que esta práctica es para los coordinadores de la asignatura, quienes van a ejecutar este proyecto en avignon, se supone que ya tienen configurado cmake, make, gcc, openmp y mpi correctamente.

## Instrucciones de ejecución
Con cmake, se puede preparar el entorno para que la llamada a make produzca todos los ejecutables que se verán en [descripción de ejecutables creados](#descripción-de-ejecutables-creados).

### Pasos

1. Preparamos el entorno
    ```
        cmake CMakeLists.txt
    ```
2. Compilamos
    ```
        make
    ```
3. Ejecutamos el ejecutable deseado con el número de nodos y procesos
    ```
        salloc -N<número de nodos> srun -n<número de procesos> ./<nombre del ejecutable deseado>
    ```



## Descripción de ejecutables creados
