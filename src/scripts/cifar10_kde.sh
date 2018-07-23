#!/usr/bin/env bash

xp_dir=../log/$1
seed=$2
cifar10_normal=$3
cifar10_outlier=$4

mkdir $xp_dir;

# CIFAR-10 training
python baseline_kde.py --xp_dir $xp_dir --dataset cifar10 --kernel gaussian --seed $seed --GridSearchCV 1 \
    --out_frac 0 --unit_norm_used l1 --pca 1 --gcn 1 --cifar10_val_frac 0 --cifar10_normal $cifar10_normal \
    --cifar10_outlier $cifar10_outlier;
