#!/usr/bin/env bash

xp_dir=../log/$1
seed=$2
contamination=$3
pca=$4
mnist_normal=$5
mnist_outlier=$6

mkdir $xp_dir;

# MNIST training
python baseline_isoForest.py --xp_dir $xp_dir --dataset mnist --n_estimators 100 --max_samples 256 \
    --contamination $contamination --out_frac 0 --seed $seed --unit_norm_used l1 --gcn 1 --pca $pca \
    --mnist_val_frac 0 --mnist_normal $mnist_normal --mnist_outlier $mnist_outlier;
