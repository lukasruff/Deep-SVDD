#!/usr/bin/env bash

device=$1
xp_dir=../log/$2
seed=$3
solver=$4
lr=$5
n_epochs=$6
nu=$7
hard_margin=$8
center_fixed=$9
block_coordinate=${10}
in_name=${11}
cifar10_architecture=${12}
cifar10_normal=${13}
cifar10_outlier=${14}


mkdir $xp_dir;

# CIFAR-10 training
python baseline.py --dataset cifar10 --solver $solver --loss svdd --lr $lr --lr_drop 1 --lr_drop_in_epoch 50 \
    --seed $seed --lr_drop_factor 10 --block_coordinate $block_coordinate --center_fixed $center_fixed \
    --use_batch_norm 1 --pretrain 0 --in_name $in_name --batch_size 200 --n_epochs $n_epochs --device $device \
    --xp_dir $xp_dir --leaky_relu 1 --weight_decay 1 --C 1e6 --reconstruction_penalty 0 --c_mean_init 1 \
    --hard_margin $hard_margin --nu $nu --out_frac 0 --weight_dict_init 0 --unit_norm_used l1 --gcn 1 --cifar10_bias 0 \
    --cifar10_rep_dim 128 --cifar10_architecture $cifar10_architecture --cifar10_normal $cifar10_normal \
    --cifar10_outlier $cifar10_outlier --nnet_diagnostics 0 --e1_diagnostics 0;
