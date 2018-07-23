import os
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from datasets.main import load_dataset
from utils.log import AD_Log
from utils.pickle import dump_isoForest, load_isoForest


class IsoForest(object):

    def __init__(self, dataset, n_estimators=100, max_samples='auto', contamination=0.1, **kwargs):

        # load dataset
        load_dataset(self, dataset)

        # initialize
        self.isoForest = None
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.initialize_isoForest(seed=self.data.seed, **kwargs)

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

    def initialize_isoForest(self, seed=0, **kwargs):

        self.isoForest = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                         contamination=self.contamination, n_jobs=-1, random_state=seed, **kwargs)

    def load_data(self, data_loader=None, pretrain=False):

        self.data = data_loader()

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def train(self):

        if self.data._X_train.ndim > 2:
            X_train_shape = self.data._X_train.shape
            X_train = self.data._X_train.reshape(X_train_shape[0], -1)
        else:
            X_train = self.data._X_train

        print("Starting training...")
        self.start_clock()

        self.isoForest.fit(X_train.astype(np.float32))

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

        scores = (-1.0) * self.isoForest.decision_function(X.astype(np.float32))  # compute anomaly score
        y_pred = (self.isoForest.predict(X.astype(np.float32)) == -1) * 1  # get prediction

        self.diag[which_set]['scores'][:, 0] = scores.flatten()
        self.diag[which_set]['acc'][0] = 100.0 * sum(y == y_pred) / len(y)

        if sum(y) > 0:
            auc = roc_auc_score(y, scores.flatten())
            self.diag[which_set]['auc'][0] = auc

        self.stop_clock()
        if which_set == 'test':
            self.test_time = self.clocked

    def dump_model(self, filename=None):

        dump_isoForest(self, filename)

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        load_isoForest(self, filename)

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
