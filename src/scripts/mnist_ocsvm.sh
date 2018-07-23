#!/usr/bin/env bash

xp_dir=../log/$1
seed=$2
nu=$3
pca=$4
mnist_normal=$5
mnist_outlier=$6

mkdir $xp_dir;

# MNIST training
python baseline_ocsvm.py --xp_dir $xp_dir --dataset mnist --loss OneClassSVM --seed $seed --kernel rbf --nu $nu \
    --GridSearchCV 1 --out_frac 0 --unit_norm_used l1 --pca $pca --mnist_val_frac 0 --mnist_normal $mnist_normal \
    --mnist_outlier $mnist_outlier;
