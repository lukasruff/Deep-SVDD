import argparse
import os
import sys
import theano

from neuralnet import NeuralNet
from config import Configuration as Cfg
from utils.log import log_exp_config, log_NeuralNet, log_AD_results
from utils.visualization.diagnostics_plot import plot_diagnostics, plot_ae_diagnostics
from utils.visualization.filters_plot import plot_filters
from utils.visualization.images_plot import plot_outliers_and_most_normal


# ====================================================================
# Parse arguments
# --------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    help="dataset name",
                    type=str, choices=["mnist", "cifar10", "gtsrb"])
parser.add_argument("--solver",
                    help="solver", type=str,
                    choices=["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam", "adamax"])
parser.add_argument("--loss",
                    help="loss function",
                    type=str, choices=["ce", "svdd", "autoencoder"])
parser.add_argument("--lr",
                    help="initial learning rate",
                    type=float)
parser.add_argument("--lr_decay",
                    help="specify if learning rate should be decayed",
                    type=int, default=0)
parser.add_argument("--lr_decay_after_epoch",
                    help="specify the epoch after learning rate should decay",
                    type=int, default=10)
parser.add_argument("--lr_drop",
                    help="specify if learning rate should drop in a specified epoch",
                    type=int, default=0)
parser.add_argument("--lr_drop_in_epoch",
                    help="specify the epoch in which learning rate should drop",
                    type=int, default=50)
parser.add_argument("--lr_drop_factor",
                    help="specify the factor by which the learning rate should drop",
                    type=int, default=10)
parser.add_argument("--momentum",
                    help="momentum rate if optimization with momentum",
                    type=float, default=0.9)
parser.add_argument("--block_coordinate",
                    help="specify if radius R and center c (if c is not fixed) should be solved for via the dual",
                    type=int, default=0)
parser.add_argument("--k_update_epochs",
                    help="update R and c in block coordinate descent only every k iterations",
                    type=int, default=5)
parser.add_argument("--center_fixed",
                    help="specify if center c should be fixed or not",
                    type=int, default=1)
parser.add_argument("--R_update_solver",
                    help="Solver for solving R",
                    type=str,
                    choices=["minimize_scalar", "lp"],
                    default="minimize_scalar")
parser.add_argument("--R_update_scalar_method",
                    help="Optimization method if minimize_scalar for solving R",
                    type=str,
                    choices=["brent", "bounded", "golden"],
                    default="bounded")
parser.add_argument("--R_update_lp_obj",
                    help="Objective used for searching R in a block coordinate descent via LP (primal or dual)",
                    type=str,
                    choices=["primal", "dual"],
                    default="primal")
parser.add_argument("--warm_up_n_epochs",
                    help="specify the first epoch the QP solver should be applied",
                    type=int, default=10)
parser.add_argument("--use_batch_norm",
                    help="specify if Batch Normalization should be applied in the network",
                    type=int, default=0)
parser.add_argument("--pretrain",
                    help="specify if weights should be pre-trained via autoenc",
                    type=int, default=0)
parser.add_argument("--nnet_diagnostics",
                    help="specify if diagnostics should be captured (faster training without)",
                    type=int, default=1)
parser.add_argument("--e1_diagnostics",
                    help="specify if diagnostics of first epoch per batch should be captured",
                    type=int, default=1)
parser.add_argument("--ae_diagnostics",
                    help="specify if diagnostics should be captured in autoencoder (faster training without)",
                    type=int, default=1)
parser.add_argument("--ae_loss",
                    help="specify the reconstruction loss of the autoencoder",
                    type=str, default="l2")
parser.add_argument("--ae_lr_drop",
                    help="specify if learning rate should drop in a specified epoch",
                    type=int, default=0)
parser.add_argument("--ae_lr_drop_in_epoch",
                    help="specify the epoch in which learning rate should drop",
                    type=int, default=50)
parser.add_argument("--ae_lr_drop_factor",
                    help="specify the factor by which the learning rate should drop",
                    type=int, default=10)
parser.add_argument("--ae_weight_decay",
                    help="specify if weight decay should be used in pretrain",
                    type=int, default=1)
parser.add_argument("--ae_C",
                    help="regularization hyper-parameter in pretrain",
                    type=float, default=1e3)
parser.add_argument("--batch_size",
                    help="batch size",
                    type=int, default=200)
parser.add_argument("--n_epochs",
                    help="number of epochs",
                    type=int)
parser.add_argument("--save_at",
                    help="number of epochs before saving model",
                    type=int, default=0)
parser.add_argument("--device",
                    help="Computation device to use for experiment",
                    type=str, default="cpu")
parser.add_argument("--xp_dir",
                    help="directory for the experiment",
                    type=str)
parser.add_argument("--in_name",
                    help="name for inputs of experiment",
                    type=str, default="")
parser.add_argument("--out_name",
                    help="name for outputs of experiment",
                    type=str, default="")
parser.add_argument("--leaky_relu",
                    help="specify if ReLU layer should be leaky",
                    type=int, default=1)
parser.add_argument("--weight_decay",
                    help="specify if weight decay should be used",
                    type=int, default=0)
parser.add_argument("--C",
                    help="regularization hyper-parameter",
                    type=float, default=1e3)
parser.add_argument("--reconstruction_penalty",
                    help="specify if a reconstruction (autoencoder) penalty should be used",
                    type=int, default=0)
parser.add_argument("--C_rec",
                    help="reconstruction (autoencoder) penalty hyperparameter",
                    type=float, default=1e3)
parser.add_argument("--dropout",
                    help="specify if dropout layers should be applied",
                    type=int, default=0)
parser.add_argument("--dropout_arch",
                    help="specify if dropout architecture should be used",
                    type=int, default=0)
parser.add_argument("--c_mean_init",
                    help="specify if center c should be initialized as mean",
                    type=int, default=1)
parser.add_argument("--c_mean_init_n_batches",
                    help="from how many batches should the mean be computed?",
                    type=int, default=-1)  # default=-1 means "all"
parser.add_argument("--hard_margin",
                    help="Train deep SVDD with hard-margin algorithm",
                    type=int, default=0)
parser.add_argument("--nu",
                    help="nu parameter in one-class SVM",
                    type=float, default=0.1)
parser.add_argument("--out_frac",
                    help="fraction of outliers in data set",
                    type=float, default=0)
parser.add_argument("--seed",
                    help="numpy seed",
                    type=int, default=0)
parser.add_argument("--ad_experiment",
                    help="specify if experiment should be two- or multiclass",
                    type=int, default=1)
parser.add_argument("--weight_dict_init",
                    help="initialize first layer filters by dictionary",
                    type=int, default=0)
parser.add_argument("--pca",
                    help="apply pca in preprocessing",
                    type=int, default=0)
parser.add_argument("--unit_norm_used",
                    help="norm to use for scaling the data to unit norm",
                    type=str, default="l2")
parser.add_argument("--gcn",
                    help="apply global contrast normalization in preprocessing",
                    type=int, default=0)
parser.add_argument("--zca_whitening",
                    help="specify if data should be whitened",
                    type=int, default=0)
parser.add_argument("--mnist_val_frac",
                    help="specify the fraction the validation set of the initial training data should be",
                    type=float, default=1./6)
parser.add_argument("--mnist_bias",
                    help="specify if bias terms are used in MNIST network",
                    type=int, default=1)
parser.add_argument("--mnist_rep_dim",
                    help="specify the dimensionality of the last layer",
                    type=int, default=16)
parser.add_argument("--mnist_architecture",
                    help="specify which network architecture should be used",
                    type=int, default=1)
parser.add_argument("--mnist_normal",
                    help="specify normal class in MNIST",
                    type=int, default=0)
parser.add_argument("--mnist_outlier",
                    help="specify outlier class in MNIST",
                    type=int, default=1)
parser.add_argument("--cifar10_bias",
                    help="specify if bias terms are used in CIFAR-10 network",
                    type=int, default=1)
parser.add_argument("--cifar10_rep_dim",
                    help="specify the dimensionality of the last layer",
                    type=int, default=32)
parser.add_argument("--cifar10_architecture",
                    help="specify which network architecture should be used",
                    type=int, default=1)
parser.add_argument("--cifar10_normal",
                    help="specify normal class in CIFAR-10",
                    type=int, default=0)
parser.add_argument("--cifar10_outlier",
                    help="specify outlier class in CIFAR-10",
                    type=int, default=1)
parser.add_argument("--gtsrb_rep_dim",
                    help="specify the dimensionality of the last layer",
                    type=int, default=32)

# ====================================================================


def main():

    args = parser.parse_args()
    print('Options:')
    for (key, value) in vars(args).iteritems():
        print("{:16}: {}".format(key, value))

    assert os.path.exists(args.xp_dir)

    # default value for basefile: string basis for all exported file names
    if args.out_name:
        base_file = "{}/{}".format(args.xp_dir, args.out_name)
    else:
        base_file = "{}/{}_{}_{}".format(args.xp_dir, args.dataset, args.solver, args.loss)

    # if pickle file already there, consider run already done
    if (os.path.exists("{}_weights.p".format(base_file)) and os.path.exists("{}_results.p".format(base_file))):
        sys.exit()

    # computation device
    if 'gpu' in args.device:
        theano.sandbox.cuda.use(args.device)

    # set save_at to n_epochs if not provided
    save_at = args.n_epochs if not args.save_at else args.save_at

    save_to = "{}_weights.p".format(base_file)
    weights = "../log/{}.p".format(args.in_name) if args.in_name else None

    # update config data

    # plot parameters
    Cfg.xp_path = args.xp_dir

    # dataset
    Cfg.seed = args.seed
    Cfg.out_frac = args.out_frac
    Cfg.ad_experiment = bool(args.ad_experiment)
    Cfg.weight_dict_init = bool(args.weight_dict_init)
    Cfg.pca = bool(args.pca)
    Cfg.unit_norm_used = args.unit_norm_used
    Cfg.gcn = bool(args.gcn)
    Cfg.zca_whitening = bool(args.zca_whitening)
    Cfg.mnist_val_frac = args.mnist_val_frac
    Cfg.mnist_bias = bool(args.mnist_bias)
    Cfg.mnist_rep_dim = args.mnist_rep_dim
    Cfg.mnist_architecture = args.mnist_architecture
    Cfg.mnist_normal = args.mnist_normal
    Cfg.mnist_outlier = args.mnist_outlier
    Cfg.cifar10_bias = bool(args.cifar10_bias)
    Cfg.cifar10_rep_dim = args.cifar10_rep_dim
    Cfg.cifar10_architecture = args.cifar10_architecture
    Cfg.cifar10_normal = args.cifar10_normal
    Cfg.cifar10_outlier = args.cifar10_outlier
    Cfg.gtsrb_rep_dim = args.gtsrb_rep_dim

    # neural network
    Cfg.softmax_loss = (args.loss == 'ce')
    Cfg.svdd_loss = (args.loss == 'svdd')
    Cfg.reconstruction_loss = (args.loss == 'autoencoder')
    Cfg.use_batch_norm = bool(args.use_batch_norm)
    Cfg.learning_rate.set_value(args.lr)
    Cfg.lr_decay = bool(args.lr_decay)
    Cfg.lr_decay_after_epoch = args.lr_decay_after_epoch
    Cfg.lr_drop = bool(args.lr_drop)
    Cfg.lr_drop_in_epoch = args.lr_drop_in_epoch
    Cfg.lr_drop_factor = args.lr_drop_factor
    Cfg.momentum.set_value(args.momentum)
    if args.solver == "rmsprop":
        Cfg.rho.set_value(0.9)
    if args.solver == "adadelta":
        Cfg.rho.set_value(0.95)
    Cfg.block_coordinate = bool(args.block_coordinate)
    Cfg.k_update_epochs = args.k_update_epochs
    Cfg.center_fixed = bool(args.center_fixed)
    Cfg.R_update_solver = args.R_update_solver
    Cfg.R_update_scalar_method = args.R_update_scalar_method
    Cfg.R_update_lp_obj = args.R_update_lp_obj
    Cfg.warm_up_n_epochs = args.warm_up_n_epochs
    Cfg.batch_size = args.batch_size
    Cfg.leaky_relu = bool(args.leaky_relu)

    # Pre-training and autoencoder configuration
    Cfg.pretrain = bool(args.pretrain)
    Cfg.ae_loss = args.ae_loss
    Cfg.ae_lr_drop = bool(args.ae_lr_drop)
    Cfg.ae_lr_drop_in_epoch = args.ae_lr_drop_in_epoch
    Cfg.ae_lr_drop_factor = args.ae_lr_drop_factor
    Cfg.ae_weight_decay = bool(args.ae_weight_decay)
    Cfg.ae_C.set_value(args.ae_C)

    # SVDD parameters
    Cfg.nu.set_value(args.nu)
    Cfg.c_mean_init = bool(args.c_mean_init)
    if args.c_mean_init_n_batches == -1:
        Cfg.c_mean_init_n_batches = "all"
    else:
        Cfg.c_mean_init_n_batches = args.c_mean_init_n_batches
    Cfg.hard_margin = bool(args.hard_margin)

    # regularization
    Cfg.weight_decay = bool(args.weight_decay)
    Cfg.C.set_value(args.C)
    Cfg.reconstruction_penalty = bool(args.reconstruction_penalty)
    Cfg.C_rec.set_value(args.C_rec)
    Cfg.dropout = bool(args.dropout)
    Cfg.dropout_architecture = bool(args.dropout_arch)

    # diagnostics
    Cfg.nnet_diagnostics = bool(args.nnet_diagnostics)
    Cfg.e1_diagnostics = bool(args.e1_diagnostics)
    Cfg.ae_diagnostics = bool(args.ae_diagnostics)

    # train
    nnet = NeuralNet(dataset=args.dataset, use_weights=weights, pretrain=Cfg.pretrain)
    # pre-train weights via autoencoder, if specified
    if Cfg.pretrain:
        nnet.pretrain(solver="adam", lr=0.0001, n_epochs=150)

    nnet.train(solver=args.solver, n_epochs=args.n_epochs, save_at=save_at, save_to=save_to)

    # pickle/serialize AD results
    if Cfg.ad_experiment:
        nnet.log_results(filename=Cfg.xp_path + "/AD_results.p")

    # text log
    nnet.log.save_to_file("{}_results.p".format(base_file))  # save log
    log_exp_config(Cfg.xp_path, args.dataset)
    log_NeuralNet(Cfg.xp_path, args.loss, args.solver, args.lr, args.momentum, None, args.n_epochs, args.C, args.C_rec,
                  args.nu)
    if Cfg.ad_experiment:
        log_AD_results(Cfg.xp_path, nnet)

    # plot diagnostics
    if Cfg.nnet_diagnostics:
        # common suffix for plot titles
        str_lr = "lr = " + str(args.lr)
        C = int(args.C)
        if not Cfg.weight_decay:
            C = None
        str_C = "C = " + str(C)
        Cfg.title_suffix = "(" + args.solver + ", " + str_C + ", " + str_lr + ")"

        if args.loss == 'autoencoder':
            plot_ae_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix)
        else:
            plot_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix)

    plot_filters(nnet, Cfg.xp_path, Cfg.title_suffix)

    # If AD experiment, plot most anomalous and most normal
    if Cfg.ad_experiment:
        n_img = 32
        plot_outliers_and_most_normal(nnet, n_img, Cfg.xp_path)


if __name__ == '__main__':
    main()
