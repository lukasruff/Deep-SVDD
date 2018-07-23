#!/usr/bin/env bash

xp_dir=../log/$1
seed=$2
pca=$3
mnist_normal=$4
mnist_outlier=$5

mkdir $xp_dir;

# MNIST training
python baseline_kde.py --xp_dir $xp_dir --dataset mnist --kernel gaussian --seed $seed --GridSearchCV 1 \
    --out_frac 0 --unit_norm_used l1 --pca $pca --mnist_val_frac 0 --mnist_normal $mnist_normal \
    --mnist_outlier $mnist_outlier;
