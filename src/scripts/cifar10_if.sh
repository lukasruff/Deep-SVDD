#!/usr/bin/env bash

xp_dir=../log/$1
seed=$2
contamination=$3
pca=$4
cifar10_normal=$5
cifar10_outlier=$6

mkdir $xp_dir;

# Cifar-10 training
python baseline_isoForest.py --xp_dir $xp_dir --dataset cifar10 --n_estimators 100 --max_samples 256 \
    --contamination $contamination --out_frac 0 --seed $seed --unit_norm_used l1 --gcn 1 --pca $pca \
    --cifar10_val_frac 0 --cifar10_normal $cifar10_normal --cifar10_outlier $cifar10_outlier;
