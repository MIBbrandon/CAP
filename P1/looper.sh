#!/bin/bash

for i in {2..16}; do
  echo "$i cores"
  for j in {0.1,0.25,0.5,0.75}; do
    echo "$j proportion of data"
    python p1hpc.py "$i" "$j"
    done
done
