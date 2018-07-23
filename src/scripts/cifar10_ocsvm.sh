#!/usr/bin/env bash

xp_dir=../log/$1
seed=$2
nu=$3
cifar10_normal=$4
cifar10_outlier=$5

mkdir $xp_dir;

# CIFAR-10 training
python baseline_ocsvm.py --xp_dir $xp_dir --dataset cifar10 --loss OneClassSVM --seed $seed --kernel rbf --nu $nu \
    --GridSearchCV 1 --out_frac 0 --unit_norm_used l1 --pca 1 --gcn 1 --cifar10_val_frac 0 \
    --cifar10_normal $cifar10_normal --cifar10_outlier $cifar10_outlier;
