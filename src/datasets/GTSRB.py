from datasets.base import DataLoader
from datasets.preprocessing import center_data, normalize_data, rescale_to_unit_interval, \
    global_contrast_normalization, zca_whitening, extract_norm_and_out, learn_dictionary, pca
from utils.visualization.mosaic_plot import plot_mosaic
from utils.misc import flush_last_line
from config import Configuration as Cfg

import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv


class GTSRB_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "gtsrb"

        # GTSRB stop sign images (class 14)
        self.n_train = 780
        self.n_val = 0
        self.n_test = 290  # 270 normal examples and 20 adversarial examples

        self.seed = Cfg.seed

        self.n_classes = 2

        self.data_path = "../data/data_gtsrb/"

        self.on_memory = True
        Cfg.store_on_gpu = True

        # load data from disk
        self.load_data()

    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu

    def load_data(self, original_scale=False):

        print("Loading data...")

        # get train data
        X = readTrafficSigns(rootpath=self.data_path, which_set="train", label=14)

        # get (normal) test data
        # X_test_norm = readTrafficSigns(rootpath=self.data_path, which_set="test", label=14)
        # sub-sample test set data of size
        np.random.seed(self.seed)
        perm = np.random.permutation(len(X))
        X_test_norm = X[perm[:100], ...]
        self._X_train = X[perm[100:], ...]
        self.n_train = len(self._X_train)
        self._y_train = np.zeros(self.n_train, dtype=np.uint8)

        # load (adversarial) test data
        X_test_adv =  np.load(self.data_path + "/Images_150.npy")
        labels_adv = np.load(self.data_path + "/Labels_150.npy")

        self._X_test = np.concatenate((X_test_norm, X_test_adv[labels_adv == 1]), axis=0).astype(np.float32)
        self._y_test = np.concatenate((np.zeros(len(X_test_norm), dtype=np.uint8),
                                       np.ones(int(np.sum(labels_adv)), dtype=np.uint8)), axis=0)
        self.n_test = len(self._X_test)

        # since val set is referenced at some points initialize empty np arrays
        self._X_val = np.empty(shape=(0, 3, 32, 32), dtype=np.float32)
        self._y_val = np.empty(shape=(0), dtype=np.uint8)

        # Adjust number of batches
        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        # shuffle
        np.random.seed(self.seed)
        perm_train = np.random.permutation(self.n_train)
        perm_test = np.random.permutation(self.n_test)
        self._X_train = self._X_train[perm_train, ...]
        self._y_train = self._y_train[perm_train]
        self._X_test = self._X_test[perm_test, ...]
        self._y_test = self._y_test[perm_test]

        # Adjust number of batches
        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

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

        if Cfg.weight_dict_init & (not nnet.pretrained):
            # initialize first layer filters by atoms of a dictionary
            W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=500)
            plot_mosaic(W1_init, title="First layer filters initialization", canvas="black",
                        export_pdf=(Cfg.xp_path + "/filters_init"))
        else:
            W1_init = None

        # Build LeNet 5 type architecture

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

        nnet.addDenseLayer(num_units=64, b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()

        nnet.addDenseLayer(num_units=Cfg.gtsrb_rep_dim, b=None)

        if Cfg.softmax_loss:
            nnet.addDenseLayer(num_units=1)
            nnet.addSigmoidLayer()
        elif Cfg.svdd_loss:
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
        else:
            raise ValueError("No valid choice of loss for dataset " + self.dataset_name)

    def build_autoencoder(self, nnet):

        if Cfg.weight_dict_init & (not nnet.pretrained):
            # initialize first layer filters by atoms of a dictionary
            W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=500)
            plot_mosaic(W1_init, title="First layer filters initialization", canvas="black",
                        export_pdf=(Cfg.xp_path + "/filters_init"))
        else:
            W1_init = None

        # Build autoencoder

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
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addDenseLayer(num_units=64, b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()

        # Code Layer
        nnet.addDenseLayer(num_units=Cfg.gtsrb_rep_dim, b=None)
        nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer

        nnet.addDenseLayer(num_units=64, b=None)
        nnet.addReshapeLayer(shape=([0], 1, 8, 8))
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
        nnet.addUpscale(scale_factor=(2, 2))

        nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=3, filter_size=(5, 5), pad='same', b=None)
        nnet.addSigmoidLayer()


def readTrafficSigns(rootpath, which_set="train", label=14):
    '''
    Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    '''

    images = [] # images
    labels = [] # corresponding labels

    if which_set == "train":
        dir_path = rootpath + "Final_Training/Images"
        prefix = dir_path + '/' + format(label, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(label, '05d') + '.csv')  # annotations file
    if which_set == "test":
        dir_path = rootpath + "Final_Test/Images"
        prefix = dir_path + '/'
        gtFile = open(prefix + '/' + 'GT-final_test.csv')  # annotations file

    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    gtReader.next() # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        x1 = int(row[3])
        y1 = int(row[4])
        x2 = int(row[5])
        y2 = int(row[6])
        img = plt.imread(prefix + row[0])  # the 1th column is the filename
        img = img[x1:x2, y1:y2, :]  # remove border of 10% around sign
        img = cv2.resize(img, (32, 32))  # resize to 32x32
        img = np.rollaxis(img, 2)  # img.shape = (3, 32, 32)
        images.append(img)
        labels.append(int(row[7]))  # the 8th column is the label
    gtFile.close()

    # convert to numpy arrays
    idx = (np.array(labels) == label)
    n = np.sum(idx)
    X = np.zeros((n, 3, 32, 32), np.float32)
    i = 0
    for img in range(len(images)):
        if idx[img]:
            X[i, :] = images[img]
            i += 1
        else:
            pass

    return X
