#!/usr/bin/env bash

mkdir ../log/mnist;
mkdir ../log/mnist/deepSVDD;

for exp in $(seq 0 9);
  do
    mkdir ../log/mnist/deepSVDD/${exp}vsall;
  done

sh experiments/mnist_svdd_exp_seeds.sh 0 &
sh experiments/mnist_svdd_exp_seeds.sh 1 &
sh experiments/mnist_svdd_exp_seeds.sh 2 &
sh experiments/mnist_svdd_exp_seeds.sh 3 &
sh experiments/mnist_svdd_exp_seeds.sh 4 &
sh experiments/mnist_svdd_exp_seeds.sh 5 &
sh experiments/mnist_svdd_exp_seeds.sh 6 &
sh experiments/mnist_svdd_exp_seeds.sh 7 &
sh experiments/mnist_svdd_exp_seeds.sh 8 &
sh experiments/mnist_svdd_exp_seeds.sh 9 &

wait
echo all experiments complete
