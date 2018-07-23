from datasets.base import DataLoader
from datasets.preprocessing import center_data, normalize_data, rescale_to_unit_interval, \
    global_contrast_normalization, zca_whitening, extract_norm_and_out, learn_dictionary, pca
from utils.visualization.mosaic_plot import plot_mosaic
from utils.misc import flush_last_line
from config import Configuration as Cfg

import os
import numpy as np
import cPickle as pickle


class CIFAR_10_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "cifar10"

        self.n_train = 45000
        self.n_val = 5000
        self.n_test = 10000

        self.seed = Cfg.seed

        if Cfg.ad_experiment:
            self.n_classes = 2
        else:
            self.n_classes = 10

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.data_path = "../data/cifar-10-batches-py/"

        self.on_memory = True
        Cfg.store_on_gpu = True

        # load data from disk
        self.load_data()

    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu

    def load_data(self, original_scale=False):

        print("Loading data...")

        # load training data
        X, y = [], []
        count = 1
        filename = '%s/data_batch_%i' % (self.data_path, count)
        while os.path.exists(filename):
            with open(filename, 'rb') as f:
                batch = pickle.load(f)
            X.append(batch['data'])
            y.append(batch['labels'])
            count += 1
            filename = '%s/data_batch_%i' % (self.data_path, count)

        # reshape data and cast them properly
        X = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32)
        y = np.concatenate(y).astype(np.int32)

        # load test set
        path = '%s/test_batch' % self.data_path
        with open(path, 'rb') as f:
            batch = pickle.load(f)

        # reshaping and casting for test data
        X_test = batch['data'].reshape(-1, 3, 32, 32).astype(np.float32)
        y_test = np.array(batch['labels'], dtype=np.int32)

        if Cfg.ad_experiment:

            # set normal and anomalous class
            normal = []
            outliers = []

            if Cfg.cifar10_normal == -1:
                normal = range(0, 10)
                normal.remove(Cfg.cifar10_outlier)
            else:
                normal.append(Cfg.cifar10_normal)

            if Cfg.cifar10_outlier == -1:
                outliers = range(0, 10)
                outliers.remove(Cfg.cifar10_normal)
            else:
                outliers.append(Cfg.cifar10_outlier)

            # extract normal and anomalous class
            X_norm, X_out, y_norm, y_out = extract_norm_and_out(X, y, normal=normal, outlier=outliers)

            # reduce outliers to fraction defined
            n_norm = len(y_norm)
            n_out = int(np.ceil(Cfg.out_frac * n_norm / (1 - Cfg.out_frac)))

            # shuffle to obtain random validation splits
            np.random.seed(self.seed)
            perm_norm = np.random.permutation(len(y_norm))
            perm_out = np.random.permutation(len(y_out))

            # split into training and validation set
            n_norm_split = int(Cfg.cifar10_val_frac * n_norm)
            n_out_split = int(Cfg.cifar10_val_frac * n_out)
            self._X_train = np.concatenate((X_norm[perm_norm[n_norm_split:]],
                                            X_out[perm_out[:n_out][n_out_split:]]))
            self._y_train = np.append(y_norm[perm_norm[n_norm_split:]],
                                      y_out[perm_out[:n_out][n_out_split:]])
            self._X_val = np.concatenate((X_norm[perm_norm[:n_norm_split]],
                                          X_out[perm_out[:n_out][:n_out_split]]))
            self._y_val = np.append(y_norm[perm_norm[:n_norm_split]],
                                    y_out[perm_out[:n_out][:n_out_split]])

            # shuffle data (since batches are extracted block-wise)
            self.n_train = len(self._y_train)
            self.n_val = len(self._y_val)
            perm_train = np.random.permutation(self.n_train)
            perm_val = np.random.permutation(self.n_val)
            self._X_train = self._X_train[perm_train]
            self._y_train = self._y_train[perm_train]
            self._X_val = self._X_train[perm_val]
            self._y_val = self._y_train[perm_val]

            # Subset train set such that we only get batches of the same size
            self.n_train = (self.n_train / Cfg.batch_size) * Cfg.batch_size
            subset = np.random.choice(len(self._X_train), self.n_train, replace=False)
            self._X_train = self._X_train[subset]
            self._y_train = self._y_train[subset]

            # Adjust number of batches
            Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

            # test set
            X_norm, X_out, y_norm, y_out = extract_norm_and_out(X_test, y_test, normal=normal, outlier=outliers)
            self._X_test = np.concatenate((X_norm, X_out))
            self._y_test = np.append(y_norm, y_out)
            perm_test = np.random.permutation(len(self._y_test))
            self._X_test = self._X_test[perm_test]
            self._y_test = self._y_test[perm_test]
            self.n_test = len(self._y_test)

        else:
            # split into training and validation sets with stored seed
            np.random.seed(self.seed)
            perm = np.random.permutation(len(X))

            self._X_train = X[perm[self.n_val:]]
            self._y_train = y[perm[self.n_val:]]
            self._X_val = X[perm[:self.n_val]]
            self._y_val = y[perm[:self.n_val]]

            self._X_test = X_test
            self._y_test = y_test

        # normalize data (if original scale should not be preserved)
        if not original_scale:

            # simple rescaling to [0,1]
            normalize_data(self._X_train, self._X_val, self._X_test, scale=np.float32(255))

            # global contrast normalization
            if Cfg.gcn:
                global_contrast_normalization(self._X_train, self._X_val, self._X_test, scale=Cfg.unit_norm_used)

            # ZCA whitening
            if Cfg.zca_whitening:
                self._X_train, self._X_val, self._X_test = zca_whitening(self._X_train, self._X_val, self._X_test)

            # rescale to [0,1] (w.r.t. min and max in train data)
            rescale_to_unit_interval(self._X_train, self._X_val, self._X_test)

            # PCA
            if Cfg.pca:
                self._X_train, self._X_val, self._X_test = pca(self._X_train, self._X_val, self._X_test, 0.95)

        flush_last_line()
        print("Data loaded.")

    def build_architecture(self, nnet):

        # implementation of different network architectures
        assert Cfg.cifar10_architecture in (1, 2, 3)

        if Cfg.cifar10_architecture == 1:

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=500)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # build architecture 1
            nnet.addInputLayer(shape=(None, 3, 32, 32))

            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same')
            else:
                if Cfg.weight_dict_init & (not nnet.pretrained):
                    nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                      W=W1_init, b=None)
                else:
                    nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                      b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5),  pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            if Cfg.cifar10_bias:
                nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim)
            else:
                nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)

            if Cfg.softmax_loss:
                nnet.addDenseLayer(num_units=1)
                nnet.addSigmoidLayer()
            elif Cfg.svdd_loss:
                nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            else:
                raise ValueError("No valid choice of loss for dataset " + self.dataset_name)

        if Cfg.cifar10_architecture == 2:

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=500)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # build architecture 2
            nnet.addInputLayer(shape=(None, 3, 32, 32))

            if Cfg.weight_dict_init & (not nnet.pretrained):
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                  W=W1_init, b=None)
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                  b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same',
                              b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same',
                              b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)

            if Cfg.softmax_loss:
                nnet.addDenseLayer(num_units=1)
                nnet.addSigmoidLayer()
            elif Cfg.svdd_loss:
                nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            else:
                raise ValueError("No valid choice of loss for dataset " + self.dataset_name)

        if Cfg.cifar10_architecture == 3:

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=32, filter_size=5, n_sample=500)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # build architecture 3
            nnet.addInputLayer(shape=(None, 3, 32, 32))

            if Cfg.weight_dict_init & (not nnet.pretrained):
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same',
                                  W=W1_init, b=None)
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same',
                                  b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same',
                              b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)

            if Cfg.softmax_loss:
                nnet.addDenseLayer(num_units=1)
                nnet.addSigmoidLayer()
            elif Cfg.svdd_loss:
                nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            else:
                raise ValueError("No valid choice of loss for dataset " + self.dataset_name)

    def build_autoencoder(self, nnet):

        # implementation of different network architectures
        assert Cfg.cifar10_architecture in (1, 2, 3)

        if Cfg.cifar10_architecture == 1:

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, 16, 5, n_sample=500)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))

            nnet.addInputLayer(shape=(None, 3, 32, 32))

            if Cfg.weight_dict_init & (not nnet.pretrained):
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                  W=W1_init, b=None)
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            # Code Layer
            nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            nnet.addReshapeLayer(shape=([0], (Cfg.cifar10_rep_dim / 4), 2, 2))
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addUpscale(scale_factor=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addUpscale(scale_factor=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addUpscale(scale_factor=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addUpscale(scale_factor=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=3, filter_size=(5, 5), pad='same', b=None)
            nnet.addSigmoidLayer()

        if Cfg.cifar10_architecture == 2:

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, 16, 5, n_sample=500)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))

            nnet.addInputLayer(shape=(None, 3, 32, 32))

            if Cfg.weight_dict_init & (not nnet.pretrained):
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                  W=W1_init, b=None)
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                  b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            # Code Layer
            nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            nnet.addReshapeLayer(shape=([0], (Cfg.cifar10_rep_dim / 16), 4, 4))
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addUpscale(scale_factor=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addUpscale(scale_factor=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addUpscale(scale_factor=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=3, filter_size=(5, 5), pad='same', b=None)
            nnet.addSigmoidLayer()

        if Cfg.cifar10_architecture == 3:

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, 32, 5, n_sample=500)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))

            nnet.addInputLayer(shape=(None, 3, 32, 32))

            if Cfg.weight_dict_init & (not nnet.pretrained):
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same',
                                  W=W1_init, b=None)
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same',
                                  b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same',
                              b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            # Code Layer
            nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            nnet.addReshapeLayer(shape=([0], (Cfg.cifar10_rep_dim / 16), 4, 4))
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addUpscale(scale_factor=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addUpscale(scale_factor=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addUpscale(scale_factor=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=3, filter_size=(5, 5), pad='same', b=None)
            nnet.addSigmoidLayer()
