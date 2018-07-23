#!/usr/bin/env bash

xp_dir=../log/$1
seed=$2
nu=$3
pca=$4

mkdir $xp_dir;

# GTSRB training
python baseline_ocsvm.py --xp_dir $xp_dir --dataset gtsrb --loss OneClassSVM --seed $seed --kernel rbf --nu $nu \
    --GridSearchCV 1 --out_frac 0 --unit_norm_used l1 --pca $pca --gcn 0;
