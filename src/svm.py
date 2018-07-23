import time
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GridSearchCV

from datasets.main import load_dataset
from kernels import degree_kernel, weighted_degree_kernel
from config import Configuration as Cfg
from utils.log import AD_Log
from utils.pickle import dump_svm, load_svm


class SVM(object):

    def __init__(self, loss, dataset, kernel, **kwargs):

        # initialize
        self.svm = None
        self.cv_svm = None
        self.loss = loss
        self.kernel = kernel
        self.K_train = None
        self.K_val = None
        self.K_test = None
        self.nu = None
        self.gamma = None
        self.initialize_svm(loss, **kwargs)

        # load dataset
        load_dataset(self, dataset)

        # train and test time
        self.clock = 0
        self.clocked = 0
        self.train_time = 0
        self.val_time = 0
        self.test_time = 0

        # Scores and AUC
        self.diag = {}

        self.diag['train'] = {}
        self.diag['val'] = {}
        self.diag['test'] = {}

        self.diag['train']['scores'] = np.zeros((len(self.data._y_train), 1))
        self.diag['val']['scores'] = np.zeros((len(self.data._y_val), 1))
        self.diag['test']['scores'] = np.zeros((len(self.data._y_test), 1))

        self.diag['train']['auc'] = np.zeros(1)
        self.diag['val']['auc'] = np.zeros(1)
        self.diag['test']['auc'] = np.zeros(1)

        self.diag['train']['acc'] = np.zeros(1)
        self.diag['val']['acc'] = np.zeros(1)
        self.diag['test']['acc'] = np.zeros(1)

        self.rho = None

        # AD results log
        self.ad_log = AD_Log()

        # diagnostics
        self.best_weight_dict = None  # attribute to reuse nnet plot-functions

    def initialize_svm(self, loss, **kwargs):

        assert loss in ('SVC', 'OneClassSVM')

        if self.kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
            kernel = self.kernel
        else:
            kernel = 'precomputed'

        if loss == 'SVC':
            self.svm = svm.SVC(kernel=kernel, C=Cfg.svm_C, **kwargs)
        if loss == 'OneClassSVM':
            self.svm = svm.OneClassSVM(kernel=kernel, nu=Cfg.svm_nu, **kwargs)
            self.cv_svm = svm.OneClassSVM(kernel=kernel, nu=Cfg.svm_nu, **kwargs)

    def load_data(self, data_loader=None, pretrain=False):

        self.data = data_loader()

    def flush_data(self):

        self.data._X_train = None
        self.data._y_train = None
        self.data._X_val = None
        self.data._y_val = None
        self.data._X_test = None
        self.data._y_test = None

        print("Data flushed from model.")

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def train(self, GridSearch=True, **kwargs):

        if self.data._X_train.ndim > 2:
            X_train_shape = self.data._X_train.shape
            X_train = self.data._X_train.reshape(X_train_shape[0], np.prod(X_train_shape[1:]))
        else:
            X_train = self.data._X_train

        print("Starting training...")
        self.start_clock()

        if self.loss == 'SVC':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set='train', **kwargs)
                self.svm.fit(self.K_train, self.data._y_train)
            else:
                self.svm.fit(X_train, self.data._y_train)

        if self.loss == 'OneClassSVM':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set='train', **kwargs)
                self.svm.fit(self.K_train)
            else:

                if GridSearch and self.kernel == 'rbf':

                    # use grid search cross-validation to select gamma
                    print("Using GridSearchCV for hyperparameter selection...")

                    # sample small hold-out set from test set for hyperparameter selection. Save as val set.
                    n_val_set = int(0.1 * self.data.n_test)
                    n_test_out = 0
                    n_test_norm = 0
                    n_val_out = 0
                    n_val_norm = 0
                    while (n_test_out == 0) | (n_test_norm == 0) | (n_val_out == 0) | (n_val_norm ==0):
                        perm = np.random.permutation(self.data.n_test)
                        self.data._X_val = self.data._X_test[perm[:n_val_set]]
                        self.data._y_val = self.data._y_test[perm[:n_val_set]]
                        # only accept small test set if AUC can be computed on val and test set
                        n_test_out = np.sum(self.data._y_test[perm[:n_val_set]])
                        n_test_norm = np.sum(self.data._y_test[perm[:n_val_set]] == 0)
                        n_val_out = np.sum(self.data._y_test[perm[n_val_set:]])
                        n_val_norm = np.sum(self.data._y_test[perm[n_val_set:]] == 0)

                    self.data._X_test = self.data._X_test[perm[n_val_set:]]
                    self.data._y_test = self.data._y_test[perm[n_val_set:]]
                    self.data.n_val = len(self.data._y_val)
                    self.data.n_test = len(self.data._y_test)

                    self.diag['val']['scores'] = np.zeros((len(self.data._y_val), 1))
                    self.diag['test']['scores'] = np.zeros((len(self.data._y_test), 1))

                    cv_auc = 0.0
                    cv_acc = 0

                    for gamma in np.logspace(-10, -1, num=10, base=2):

                        # train on selected gamma
                        self.cv_svm = svm.OneClassSVM(kernel='rbf', nu=Cfg.svm_nu, gamma=gamma)
                        self.cv_svm.fit(X_train)

                        # predict on small hold-out set
                        self.predict(which_set='val')

                        # save model if AUC on hold-out set improved
                        if self.diag['val']['auc'] > cv_auc:
                            self.svm = self.cv_svm
                            self.nu = Cfg.svm_nu
                            self.gamma = gamma
                            cv_auc = self.diag['val']['auc']
                            cv_acc = self.diag['val']['acc']

                    # save results of best cv run
                    self.diag['val']['auc'] = cv_auc
                    self.diag['val']['acc'] = cv_acc

                else:
                    # if rbf-kernel, re-initialize svm with gamma minimizing the
                    # numerical error
                    if self.kernel == 'rbf':
                        gamma = 1 / (np.max(pairwise_distances(X_train)) ** 2)
                        self.svm = svm.OneClassSVM(kernel='rbf', nu=Cfg.svm_nu, gamma=gamma)

                    self.svm.fit(X_train)

                    self.nu = Cfg.svm_nu
                    self.gamma = gamma

        self.stop_clock()
        self.train_time = self.clocked

    def predict(self, which_set='train', **kwargs):

        assert which_set in ('train', 'val', 'test')

        if which_set == 'train':
            X = self.data._X_train
            y = self.data._y_train
        if which_set == 'val':
            X = self.data._X_val
            y = self.data._y_val
        if which_set == 'test':
            X = self.data._X_test
            y = self.data._y_test

        # reshape to 2D if input is tensor
        if X.ndim > 2:
            X_shape = X.shape
            X = X.reshape(X_shape[0], np.prod(X_shape[1:]))

        print("Starting prediction...")
        self.start_clock()

        if self.loss == 'SVC':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set=which_set, **kwargs)
                if which_set == 'train':
                    scores = self.svm.decision_function(self.K_train)
                if which_set == 'test':
                    scores = self.svm.decision_function(self.K_test)
            else:
                scores = self.svm.decision_function(X)

            auc = roc_auc_score(y, scores[:, 0])

            self.diag[which_set]['scores'] = scores
            self.diag[which_set]['auc'][0] = auc

        if self.loss == 'OneClassSVM':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set=which_set, **kwargs)
                if which_set == 'train':
                    scores = (-1.0) * self.svm.decision_function(self.K_train)
                    y_pred = (self.svm.predict(self.K_train) == -1) * 1
                if which_set == 'test':
                    scores = (-1.0) * self.svm.decision_function(self.K_test)
                    y_pred = (self.svm.predict(self.K_test) == -1) * 1
            else:
                if which_set == "val":
                    scores = (-1.0) * self.cv_svm.decision_function(X)
                    y_pred = (self.cv_svm.predict(X) == -1) * 1
                else:
                    scores = (-1.0) * self.svm.decision_function(X)
                    y_pred = (self.svm.predict(X) == -1) * 1

            self.diag[which_set]['scores'][:, 0] = scores.flatten()
            self.diag[which_set]['acc'][0] = 100.0 * sum(y == y_pred) / len(y)

            if sum(y) > 0:
                auc = roc_auc_score(y, scores.flatten())
                self.diag[which_set]['auc'][0] = auc

        self.stop_clock()
        if which_set == 'test':
            self.rho = -self.svm.intercept_[0]
            self.test_time = self.clocked
        if which_set == 'val':
            self.val_time = self.clocked


    def dump_model(self, filename=None):

        dump_svm(self, filename)

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        load_svm(self, filename)

    def log_results(self, filename=None):
        """
        log the results relevant for anomaly detection
        """

        self.ad_log['train_auc'] = self.diag['train']['auc'][-1]
        self.ad_log['train_accuracy'] = self.diag['train']['acc'][-1]
        self.ad_log['train_time'] = self.train_time

        self.ad_log['val_auc'] = self.diag['val']['auc'][-1]
        self.ad_log['val_accuracy'] = self.diag['val']['acc'][-1]
        self.ad_log['val_time'] = self.val_time

        self.ad_log['test_auc'] = self.diag['test']['auc'][-1]
        self.ad_log['test_accuracy'] = self.diag['test']['acc'][-1]
        self.ad_log['test_time'] = self.test_time

        self.ad_log.save_to_file(filename=filename)


    def get_kernel_matrix(self, kernel, which_set='train', **kwargs):

        assert kernel in ('DegreeKernel', 'WeightedDegreeKernel')

        if kernel == 'DegreeKernel':
            kernel_function = degree_kernel
        if kernel == 'WeightedDegreeKernel':
            kernel_function = weighted_degree_kernel

        if which_set == 'train':
            self.K_train = kernel_function(self.data._X_train, self.data._X_train, **kwargs)
        if which_set == 'val':
            self.K_val = kernel_function(self.data._X_val, self.data._X_train, **kwargs)
        if which_set == 'test':
            self.K_test = kernel_function(self.data._X_test, self.data._X_train, **kwargs)
