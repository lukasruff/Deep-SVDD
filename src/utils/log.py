import time
import os
import numpy as np

import cPickle as pickle

from config import Configuration as Cfg


class Log(dict):

    def __init__(self, dataset_name):

        dict.__init__(self)

        self['dataset_name'] = dataset_name

        self['date_and_time'] = time.strftime('%d-%m-%Y--%H-%M-%S')

        self['time_stamp'] = []
        self['layer_tag'] = []

        self['train_objective'] = []
        self['train_accuracy'] = []
        self['train_emp_loss'] = []
        self['train_auc'] = []
        self['train_outlier_scores_summary'] = []
        self['train_normal_scores_summary'] = []
        self['train_outlier_rep_norm_summary'] = []
        self['train_normal_rep_norm_summary'] = []

        self['val_objective'] = []
        self['val_accuracy'] = []
        self['val_emp_loss'] = []
        self['val_auc'] = []
        self['val_outlier_scores_summary'] = []
        self['val_normal_scores_summary'] = []
        self['val_outlier_rep_norm_summary'] = []
        self['val_normal_rep_norm_summary'] = []

        self['test_objective'] = []
        self['test_accuracy'] = []
        self['test_emp_loss'] = []
        self['test_auc'] = []
        self['test_outlier_scores_summary'] = []
        self['test_normal_scores_summary'] = []
        self['test_outlier_rep_norm_summary'] = []
        self['test_normal_rep_norm_summary'] = []

        # self['test_objective'] = -1
        # self['test_accuracy'] = -1

        self['l2_penalty'] = []

        for key in Cfg.__dict__:
            if key.startswith('__'):
                continue
            if key not in ('C', 'D', 'learning_rate', 'momentum', 'rho', 'nu'):
                self[key] = getattr(Cfg, key)
            else:
                self[key] = getattr(Cfg, key).get_value()

    def store_architecture(self, nnet):

        self['layers'] = dict()
        for layer in nnet.all_layers:
            self['layers'][layer.name] = dict()
            if layer.isdense:
                self['layers'][layer.name]["n_in"] = \
                    np.prod(layer.input_shape[1:])
                self['layers'][layer.name]["n_out"] = layer.num_units

            if layer.isconv:
                self['layers'][layer.name]["n_filters"] = layer.num_filters
                self['layers'][layer.name]["f_size"] = layer.filter_size

            if layer.ismaxpool:
                self['layers'][layer.name]["pool_size"] = layer.pool_size

    def save_to_file(self, filename=None):

        if not filename:
            filename = '../log/all/{}-0'.format(self['date_and_time'])
            count = 1
            while os.path.exists(filename):
                filename = '../log/all/{}-{}'\
                    .format(self['date_and_time'], count)
                count += 1
            filename += '.p'

        pickle.dump(self, open(filename, 'wb'))
        print('Experiment logged in {}'.format(filename))


class AD_Log(dict):

    def __init__(self):

        dict.__init__(self)

        self['date_and_time'] = time.strftime('%d-%m-%Y--%H-%M-%S')

        self['train_auc'] = 0
        self['train_accuracy'] = 0
        self['train_time'] = 0

        self['val_auc'] = 0
        self['val_accuracy'] = 0

        self['test_auc'] = 0
        self['test_accuracy'] = 0
        self['test_time'] = 0

    def save_to_file(self, filename=None):

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        print('Anomaly detection results logged in {}'.format(filename))


def log_exp_config(xp_path, dataset):
    """
    log configuration of the experiment in a .txt-file
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("Experiment configuration\n")
    log.write("Dataset: {}\n".format(dataset))
    log.write("Seed: {}\n".format(Cfg.seed))
    log.write("Fraction of Outliers: {}\n".format(Cfg.out_frac))
    log.write("First layer weight init by dictionary: {}\n".format(Cfg.weight_dict_init))
    log.write("PCA pre-processing? {}\n".format(Cfg.pca))
    log.write("Norm used: {}\n".format(Cfg.unit_norm_used))
    log.write("Global contrast normalization? {}\n".format(Cfg.gcn))
    log.write("ZCA Whitening? {}\n".format(Cfg.zca_whitening))

    if dataset == 'mnist':
        str_normal = str(Cfg.mnist_normal)
        str_outlier = str(Cfg.mnist_outlier)
        if Cfg.mnist_normal == -1:
            str_normal = "all"
        if Cfg.mnist_outlier == -1:
            str_outlier = "all"
        log.write("MNIST classes: {} vs. {}\n".format(str_normal, str_outlier))
        log.write("MNIST representation dimensionality: {}\n".format(Cfg.mnist_rep_dim))
        log.write("MNIST architecture used: {}\n".format(Cfg.mnist_architecture))
        log.write("MNIST Network with bias terms? {}\n".format(Cfg.mnist_bias))

    if dataset == 'cifar10':
        str_normal = str(Cfg.cifar10_normal)
        str_outlier = str(Cfg.cifar10_outlier)
        if Cfg.cifar10_normal == -1:
            str_normal = "all"
        if Cfg.cifar10_outlier == -1:
            str_outlier = "all"
        log.write("CIFAR-10 classes: {} vs. {}\n".format(str_normal, str_outlier))
        log.write("CIFAR-10 representation dimensionality: {}\n".format(Cfg.cifar10_rep_dim))
        log.write("CIFAR-10 architecture used: {}\n".format(Cfg.cifar10_architecture))
        log.write("CIFAR-10 Network with bias terms? {}\n".format(Cfg.cifar10_bias))

    if dataset == 'gtsrb':
        log.write("GTSRB representation dimensionality: {}\n".format(Cfg.gtsrb_rep_dim))

    log.write("\n\n")
    log.close()


def log_NeuralNet(xp_path, loss, solver, learning_rate, momentum, rho, n_epochs, C, C_rec, nu):
    """
    log configuration of NeuralNet-class instance
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("NeuralNet configuration\n")
    log.write("Loss: {}\n".format(loss))
    log.write("Pre-training? {}\n".format(Cfg.pretrain))
    log.write("Solver: {}\n".format(solver))
    log.write("Learning rate: {}\n".format(learning_rate))
    log.write("Learning rate decay? {}\n".format(Cfg.lr_decay))
    log.write("Learning rate decay after epoch: {}\n".format(Cfg.lr_decay_after_epoch))
    log.write("Learning rate drop? {}\n".format(Cfg.lr_drop))
    log.write("Learning rate drop in epoch: {}\n".format(Cfg.lr_drop_in_epoch))
    log.write("Learning rate drop by factor: {}\n".format(Cfg.lr_drop_factor))
    log.write("Momentum: {}\n".format(momentum))
    log.write("Rho: {}\n".format(rho))
    log.write("Use Batch Normalization? {}\n".format(Cfg.use_batch_norm))
    log.write("Number of epochs: {}\n".format(n_epochs))
    log.write("Batch size: {}\n".format(Cfg.batch_size))
    log.write("Leaky ReLU: {}\n\n".format(Cfg.leaky_relu))

    log.write("Regularization\n")
    log.write("Weight decay: {}\n".format(Cfg.weight_decay))
    log.write("C-parameter: {}\n".format(C))
    log.write("Dropout: {}\n".format(Cfg.dropout))
    log.write("Dropout architecture? {}\n\n".format(Cfg.dropout_architecture))

    if Cfg.pretrain:
        log.write("Pre-Training Configuration:\n")
        log.write("Reconstruction loss: {}\n".format(Cfg.ae_loss))
        log.write("Learning rate drop? {}\n".format(Cfg.ae_lr_drop))
        log.write("Learning rate drop in epoch: {}\n".format(Cfg.ae_lr_drop_in_epoch))
        log.write("Learning rate drop by factor: {}\n".format(Cfg.ae_lr_drop_factor))
        log.write("Weight decay: {}\n".format(Cfg.ae_weight_decay))
        log.write("C-parameter: {}\n\n".format(Cfg.ae_C.get_value()))

    if loss == 'svdd':
        log.write("SVDD\n")
        log.write("Hard margin objective? {}\n".format(Cfg.hard_margin))
        log.write("Block coordinate descent used to solve R (and possibly c)? {}\n".format(Cfg.block_coordinate))
        log.write("Is center c fixed? {}\n".format(Cfg.center_fixed))
        log.write("Solver for R: {}\n".format(Cfg.R_update_solver))
        log.write("Optimization method if minimize_scalar: {}\n".format(Cfg.R_update_scalar_method))
        log.write("Objective on which R is optimized if LP: {}\n".format(Cfg.R_update_lp_obj))
        log.write("Block coordinate descent applied from epoch: {}\n".format(Cfg.warm_up_n_epochs))
        log.write("(R,c) block update every k epoch with k={}\n".format(Cfg.k_update_epochs))
        log.write("Reconstruction regularization: {}\n".format(Cfg.reconstruction_penalty))
        log.write("C_rec-parameter: {}\n".format(C_rec))
        log.write("Nu-parameter: {}\n".format(nu))
        log.write("Mean initialization of c? {}\n".format(Cfg.c_mean_init))
        log.write("Number of batches for mean initialization of c: {}\n".format(Cfg.c_mean_init_n_batches))

    if loss == 'autoencoder':
        log.write("Autoencoder\n")
        log.write("Reconstruction loss: {}\n".format(Cfg.ae_loss))
        log.write("Learning rate drop? {}\n".format(Cfg.ae_lr_drop))
        log.write("Learning rate drop in epoch: {}\n".format(Cfg.ae_lr_drop_in_epoch))
        log.write("Learning rate drop by factor: {}\n".format(Cfg.ae_lr_drop_factor))
        log.write("Weight decay: {}\n".format(Cfg.ae_weight_decay))
        log.write("C-parameter: {}\n".format(Cfg.ae_C.get_value()))

    log.write("\n\n")
    log.close()


def log_SVM(xp_path, loss, kernel, gamma, nu):
    """
    log configuration of SVM-class instance
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("SVM configuration\n")
    log.write("Loss: {}\n".format(loss))
    log.write("Kernel: {}\n".format(kernel))
    log.write("GridSearchCV for hyperparameter selection? {}\n".format(Cfg.svm_GridSearchCV))
    log.write("Gamma: {}\n".format(gamma))
    log.write("Nu-parameter: {}\n".format(nu))

    log.write("\n\n")
    log.close()


def log_KDE(xp_path, kernel, bandwidth):
    """
    log configuration of KDE-class instance
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("KDE configuration\n")
    log.write("Kernel: {}\n".format(kernel))
    log.write("Bandwidth: {}\n".format(bandwidth))
    log.write("GridSearchCV for hyperparameter selection? {}\n".format(Cfg.kde_GridSearchCV))

    log.write("\n\n")
    log.close()


def log_isoForest(xp_path, n_estimators, max_samples, contamination):
    """
    log configuration of isoForest-class instance
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("Isolation Forest configuration\n")
    log.write("Number of base estimators in the ensemble: {}\n".format(n_estimators))
    log.write("Number of samples drawn to train each base estimator: {}\n".format(max_samples))
    log.write("Expected fraction of outliers in the training set (contamination): {}\n".format(contamination))

    log.write("\n\n")
    log.close()


def log_AD_results(xp_path, learner):
    """
    log the final results to compare the performance of various learners
    """

    log_file = "{}/log.txt".format(xp_path)
    log = open(log_file, "a")

    log.write("Results\n\n")

    log.write("Train AUC: {} %\n".format(round(learner.diag['train']['auc'][-1]*100, 4)))
    log.write("Train accuracy: {} %\n".format(round(learner.diag['train']['acc'][-1], 4)))
    log.write("Train time: {}\n\n".format(round(learner.train_time, 4)))

    log.write("Val AUC: {} %\n".format(round(learner.diag['val']['auc'][-1] * 100, 4)))
    log.write("Val accuracy: {} %\n\n".format(round(learner.diag['val']['acc'][-1], 4)))

    log.write("Test AUC: {} %\n".format(round(learner.diag['test']['auc'][-1]*100, 4)))
    log.write("Test accuracy: {} %\n".format(round(learner.diag['test']['acc'][-1], 4)))
    log.write("Test time: {}\n".format(round(learner.test_time, 4)))

    log.write("\n\n")
    log.close()
