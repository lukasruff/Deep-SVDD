#!/usr/bin/env bash

exp=$1

for seed in $(seq 10);
  do
    sh scripts/mnist_svdd.sh cpu mnist/deepSVDD/${exp}vsall/seed_${seed} ${seed} adam 0.0001 150 1 0 1 $exp -1;
  done
