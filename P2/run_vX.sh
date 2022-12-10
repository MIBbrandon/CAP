#!/bin/bash

make

is_local=""
category="other"
version="vX${is_local}"
results_folder="$PWD/versions/$category/$version/results/"
if [[ "$is_local" == "_local" ]]
then
  for num_processes in {1..8}
  do
    # Para cada combinación de número de nodos y número de procesos que tenga sentido
    # creamos un CSV que guarde los tiempos

    results_file=${results_folder}/results_${version}_${num_nodes}_${num_processes}.csv

    echo "run, gray_time, hsl_time, yuv_time" > $results_file
    for run in {1..3}
    do
        echo "Run $run on $num_nodes nodes with $num_processes processes:"
        times=($(mpirun -n ${num_processes} ./contrast | grep "0 processing time" | grep -Eo "[0-9]+.[0-9]+"))
        echo "${times[0]}, ${times[1]}, ${times[2]}"
        echo "${run}, ${times[0]}, ${times[1]}, ${times[2]}" >> $results_file
    done
  done
else
  for num_nodes in 1 2 4 6
  do
      for num_processes in {1..24}
      do
          # Usar este IF para saltar a un experimento en concreto
          if ((${num_nodes} < 6)) || ((${num_processes} < 21)); then
              continue
          fi

          # # Usar este IF para parar en este experimento (NO incluido)
          # if ((${num_nodes} == 1)) && ((${num_processes} == 1)); then
          #     echo "Parando en $num_nodes nodes y $num_processes processes"
          #     exit 0
          # fi

          if ((${num_processes} <= ${num_nodes} * 4)) && ((${num_processes} > (${num_nodes} - 1) * 4))
          then
              # Para cada combinación de número de nodos y número de procesos que tenga sentido
              # (es decir, que num_processes <= num_nodes * 4, ya que cada nodo tiene 4 procesadores),
              # creamos un CSV que guarde los tiempos

              results_file=${results_folder}results_${version}_${num_nodes}_${num_processes}.csv

              echo "run, gray_time, hsl_time, yuv_time" > $results_file
              for run in {1..3}
              do
                  echo "Run $run on $num_nodes nodes with $num_processes processes:"
                  times=($(srun -N ${num_nodes} -n ${num_processes} ./contrast | grep "0 processing time" | grep -Eo "[0-9]+.[0-9]+"))
                  echo "${times[0]}, ${times[1]}, ${times[2]}"
                  echo "${run}, ${times[0]}, ${times[1]}, ${times[2]}" >> $results_file
              done
          fi
      done
  done
fi
