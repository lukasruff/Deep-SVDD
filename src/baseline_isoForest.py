import argparse
import os

from isoForest import IsoForest
from config import Configuration as Cfg
from utils.log import log_exp_config, log_isoForest, log_AD_results
from utils.visualization.images_plot import plot_outliers_and_most_normal


# ====================================================================
# Parse arguments
# --------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    help="dataset name",
                    type=str, choices=["mnist", "cifar10", "gtsrb"])
parser.add_argument("--xp_dir",
                    help="directory for the experiment",
                    type=str)
parser.add_argument("--n_estimators",
                    help="Specify the number of base estimators in the ensemble",
                    type=int, default=100)
parser.add_argument("--max_samples",
                    help="Number of samples drawn to train each base estimator",
                    type=int, default=256)
parser.add_argument("--contamination",
                    help="Expected fraction of outliers in the training set (contamination)",
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
parser.add_argument("--unit_norm_used",
                    help="norm to use for scaling the data to unit norm",
                    type=str, default="l1")
parser.add_argument("--gcn",
                    help="apply global contrast normalization in preprocessing",
                    type=int, default=0)
parser.add_argument("--zca_whitening",
                    help="specify if data should be whitened",
                    type=int, default=0)
parser.add_argument("--pca",
                    help="apply pca in preprocessing",
                    type=int, default=0)
parser.add_argument("--mnist_val_frac",
                    help="specify the fraction the validation set of the initial training data should be",
                    type=float, default=0)  # default of 0 as ensemble bootstrapping performed internally.
parser.add_argument("--mnist_normal",
                    help="specify normal class in MNIST",
                    type=int, default=0)
parser.add_argument("--mnist_outlier",
                    help="specify outlier class in MNIST",
                    type=int, default=-1)
parser.add_argument("--cifar10_val_frac",
                    help="specify the fraction the validation set of the initial training data should be",
                    type=float, default=0)  # default of 0 as ensemble bootstrapping performed internally.
parser.add_argument("--cifar10_normal",
                    help="specify normal class in CIFAR-10",
                    type=int, default=1)
parser.add_argument("--cifar10_outlier",
                    help="specify outlier class in CIFAR-10",
                    type=int, default=-1)

# ====================================================================


def main():

    args = parser.parse_args()
    print('Options:')
    for (key, value) in vars(args).iteritems():
        print("{:16}: {}".format(key, value))

    assert os.path.exists(args.xp_dir)

    # update config data

    # plot parameters
    Cfg.xp_path = args.xp_dir

    # dataset
    Cfg.seed = args.seed
    Cfg.out_frac = args.out_frac
    Cfg.ad_experiment = bool(args.ad_experiment)
    Cfg.unit_norm_used = args.unit_norm_used
    Cfg.gcn = bool(args.gcn)
    Cfg.zca_whitening = bool(args.zca_whitening)
    Cfg.pca = bool(args.pca)
    Cfg.mnist_val_frac = args.mnist_val_frac
    Cfg.mnist_normal = args.mnist_normal
    Cfg.mnist_outlier = args.mnist_outlier
    Cfg.cifar10_val_frac = args.cifar10_val_frac
    Cfg.cifar10_normal = args.cifar10_normal
    Cfg.cifar10_outlier = args.cifar10_outlier

    # initialize Isolation Forest
    model = IsoForest(dataset=args.dataset, n_estimators=args.n_estimators, max_samples=args.max_samples,
                      contamination=args.contamination)

    # train IF model
    model.train()

    # predict
    model.predict(which_set='train')
    model.predict(which_set='test')

    # log
    log_exp_config(Cfg.xp_path, args.dataset)
    log_isoForest(Cfg.xp_path, args.n_estimators, args.max_samples, args.contamination)
    log_AD_results(Cfg.xp_path, model)

    # pickle/serialize
    model.dump_model(filename=Cfg.xp_path + "/model.p")
    model.log_results(filename=Cfg.xp_path + "/AD_results.p")

    # plot targets and outliers sorted
    n_img = 32
    plot_outliers_and_most_normal(model, n_img, Cfg.xp_path)


if __name__ == '__main__':
    main()
