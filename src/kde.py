import os
import time
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import pairwise_distances

from datasets.main import load_dataset
from utils.log import AD_Log
from utils.pickle import dump_kde, load_kde


class KDE(object):

    def __init__(self, dataset, kernel, **kwargs):

        # initialize
        self.kde = None
        self.kernel = kernel
        self.bandwidth = None
        self.initialize_kde(**kwargs)

        # load dataset
        load_dataset(self, dataset)

        # train and test time
        self.clock = 0
        self.clocked = 0
        self.train_time = 0
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

        # AD results log
        self.ad_log = AD_Log()

        # diagnostics
        self.best_weight_dict = None  # attribute to reuse nnet plot-functions

    def initialize_kde(self, **kwargs):

        self.kde = KernelDensity(kernel=self.kernel, **kwargs)
        self.bandwidth = self.kde.bandwidth

    def load_data(self, data_loader=None, pretrain=False):

        self.data = data_loader()

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def train(self, bandwidth_GridSearchCV=True):

        if self.data._X_train.ndim > 2:
            X_train_shape = self.data._X_train.shape
            X_train = self.data._X_train.reshape(X_train_shape[0], -1)
        else:
            X_train = self.data._X_train

        print("Starting training...")
        self.start_clock()

        if bandwidth_GridSearchCV:
            # use grid search cross-validation to select bandwidth
            print("Using GridSearchCV for bandwidth selection...")

            # params = {'bandwidth': np.logspace(0.5, 5, num=10, base=2)}
            params = {'bandwidth': np.logspace(- 4.5, 5, num=20, base=2)}

            hyper_kde = GridSearchCV(KernelDensity(kernel=self.kernel), params, n_jobs=-1, cv=5, verbose=0)
            hyper_kde.fit(X_train)

            self.bandwidth = hyper_kde.best_estimator_.bandwidth
            self.kde = hyper_kde.best_estimator_
        else:
            # if exponential kernel, re-initialize kde with bandwidth minimizing
            # the numerical error
            if self.kernel == 'exponential':
                bandwidth = np.max(pairwise_distances(X_train)) ** 2
                self.kde = KernelDensity(kernel=self.kernel,
                                         bandwidth=bandwidth)

            self.kde.fit(X_train)

        self.stop_clock()
        self.train_time = self.clocked

    def predict(self, which_set='train'):

        assert which_set in ('train', 'test')

        if which_set == 'train':
            X = self.data._X_train
            y = self.data._y_train
        if which_set == 'test':
            X = self.data._X_test
            y = self.data._y_test

        # reshape to 2D if input is tensor
        if X.ndim > 2:
            X_shape = X.shape
            X = X.reshape(X_shape[0], -1)

        print("Starting prediction...")
        self.start_clock()

        scores = (-1.0) * self.kde.score_samples(X)  # anomaly score
        self.diag[which_set]['scores'][:, 0] = scores.flatten()

        if sum(y) > 0:
            auc = roc_auc_score(y, scores.flatten())
            self.diag[which_set]['auc'][0] = auc

        self.stop_clock()
        if which_set == 'test':
            self.test_time = self.clocked

    def dump_model(self, filename=None):

        dump_kde(self, filename)

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        load_kde(self, filename)

    def log_results(self, filename=None):
        """
        log the results relevant for anomaly detection
        """

        self.ad_log['train_auc'] = self.diag['train']['auc'][-1]
        self.ad_log['train_accuracy'] = self.diag['train']['acc'][-1]
        self.ad_log['train_time'] = self.train_time

        self.ad_log['test_auc'] = self.diag['test']['auc'][-1]
        self.ad_log['test_accuracy'] = self.diag['test']['acc'][-1]
        self.ad_log['test_time'] = self.test_time

        self.ad_log.save_to_file(filename=filename)
