#!/usr/bin/env bash

device=$1
xp_dir=../log/$2
seed=$3
solver=$4
lr=$5
n_epochs=$6
cifar10_architecture=$7
cifar10_normal=$8
cifar10_outlier=$9


mkdir $xp_dir;

# CIFAR-10 training
python baseline.py --device $device --xp_dir $xp_dir --dataset cifar10 --solver $solver --loss autoencoder --lr $lr \
    --ae_lr_drop 1 --ae_lr_drop_in_epoch 250 --ae_lr_drop_factor 10 --seed $seed --n_epochs $n_epochs --batch_size 200 \
    --use_batch_norm 1 --out_frac 0 --ae_weight_decay 1 --ae_C 1e6 --gcn 1 --unit_norm_used l1 --weight_dict_init 1 \
    --leaky_relu 1 --ae_loss l2 --cifar10_bias 0 --cifar10_rep_dim 128 --cifar10_architecture $cifar10_architecture \
    --cifar10_normal $cifar10_normal --cifar10_outlier $cifar10_outlier;
