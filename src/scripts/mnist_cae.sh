device=$1
xp_dir=../log/$2
seed=$3
solver=$4
lr=$5
n_epochs=$6
mnist_normal=$7
mnist_outlier=$8

mkdir $xp_dir;

# MNIST training
python baseline.py --device $device --xp_dir $xp_dir --dataset mnist --solver $solver --loss autoencoder --lr $lr \
    --ae_lr_drop 1 --ae_lr_drop_in_epoch 150 --ae_lr_drop_factor 10 --seed $seed --n_epochs $n_epochs --batch_size 200 \
    --use_batch_norm 1 --out_frac 0 --ae_weight_decay 1 --ae_C 1e3 --gcn 1 --unit_norm_used l1 --weight_dict_init 1 \
    --leaky_relu 1 --ae_loss l2 --mnist_bias 0 --mnist_rep_dim 32 --mnist_normal $mnist_normal \
    --mnist_outlier $mnist_outlier;
