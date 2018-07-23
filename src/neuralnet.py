import time
import os
import numpy as np
import cPickle as pickle
from lasagne.layers import InputLayer

import opt.sgd.train
import opt.sgd.updates

from opt.sgd.updates import update_R, update_R_c
from sklearn.metrics import roc_auc_score
from datasets.main import load_dataset
from utils.monitoring import performance
from utils.misc import get_five_number_summary
from utils.pickle import dump_weights, load_weights
from utils.log import Log, AD_Log
from utils.diag import NNetDataDiag, NNetParamDiag
from layers import ConvLayer, ReLU, LeakyReLU, MaxPool, Upscale, DenseLayer, BatchNorm, DropoutLayer, Dimshuffle, \
    Reshape, Sigmoid, Softmax, Norm, Abs
from config import Configuration as Cfg


class NeuralNet:

    def __init__(self, dataset, use_weights=None, pretrain=False, profile=False):
        """
        initialize instance
        """

        # whether to enable profiling in Theano functions
        self.profile = profile

        self.initialize_variables(dataset)

        # load dataset
        load_dataset(self, dataset.lower(), pretrain)

        if use_weights and not pretrain:
            self.load_weights(use_weights)

    def initialize_variables(self, dataset):

        self.all_layers, self.trainable_layers = (), ()

        self.n_conv_layers = 0
        self.n_dense_layers = 0
        self.n_relu_layers = 0
        self.n_leaky_relu_layers = 0
        self.n_bn_layers = 0
        self.n_norm_layers = 0
        self.n_abs_layers = 0
        self.n_maxpool_layers = 0
        self.n_upscale_layers = 0
        self.n_dropout_layers = 0
        self.n_dimshuffle_layers = 0
        self.n_reshape_layers = 0
        self.R_init = 0

        self.learning_rate_init = Cfg.learning_rate.get_value()

        self.it = 0
        self.clock = 0

        self.pretrained = False  # set to True after pretraining such that dictionary initialization mustn't be repeated

        self.diag = {}  # init an empty dictionary to hold diagnostics
        self.log = Log(dataset_name=dataset)
        self.ad_log = AD_Log()

        self.dense_layers, self.conv_layers, = [], []

    def compile_updates(self):
        """ create network from architecture given in modules (determined by dataset)
        create Theano compiled functions
        """

        opt.sgd.updates.create_update(self)

    def compile_autoencoder(self):
        """
        create network from autoencoder architecture (determined by dataset)
        and compile Theano update functions.
        """

        print("Compiling autoencoder...")
        opt.sgd.updates.create_autoencoder(self)
        print("Autoencoder compiled.")

    def load_data(self, data_loader=None, pretrain=False):

        self.data = data_loader()

        if pretrain:
            self.data.build_autoencoder(self)

            for layer in self.all_layers:
                setattr(self, layer.name + "_layer", layer)
        elif Cfg.reconstruction_loss:
            self.data.build_autoencoder(self)

            for layer in self.all_layers:
                setattr(self, layer.name + "_layer", layer)

            self.log.store_architecture(self)
        elif Cfg.svdd_loss and Cfg.reconstruction_penalty:
            self.data.build_autoencoder(self)

            for layer in self.all_layers:
                setattr(self, layer.name + "_layer", layer)

            self.log.store_architecture(self)
        else:
            self.data.build_architecture(self)

            for layer in self.all_layers:
                setattr(self, layer.name + "_layer", layer)

            self.log.store_architecture(self)

    def flush_data(self):

        self.data._X_train = None
        self.data._y_train = None
        self.data._X_val = None
        self.data._y_val = None
        self.data._X_test = None
        self.data._y_test = None

        print("Data flushed from network.")

    def next_layers(self, layer):

        flag = False
        for current_layer in self.all_layers:
            if flag:
                yield current_layer
            if current_layer is layer:
                flag = True

    def previous_layers(self, layer):

        flag = False
        for current_layer in reversed(self.all_layers):
            if flag:
                yield current_layer
            if current_layer is layer:
                flag = True

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def pretrain(self, solver, lr, n_epochs):
        """
        pre-train weights with an autoencoder
        """

        self.ae_solver = solver.lower()
        self.ae_learning_rate = lr
        self.ae_n_epochs = n_epochs

        # set learning rate
        lr_tmp = Cfg.learning_rate.get_value()
        Cfg.learning_rate.set_value(Cfg.floatX(lr))

        self.compile_autoencoder()

        from opt.sgd.train import train_autoencoder
        train_autoencoder(self)

        # remove layer attributes, re-initialize network and reset learning rate
        for layer in self.all_layers:
            delattr(self, layer.name + "_layer")
        self.initialize_variables(self.data.dataset_name)
        Cfg.learning_rate.set_value(Cfg.floatX(lr_tmp))
        self.pretrained = True  # set to True that dictionary initialization mustn't be repeated

        # load network architecture
        if Cfg.svdd_loss and Cfg.reconstruction_penalty:
            self.data.build_autoencoder(self)

            for layer in self.all_layers:
                setattr(self, layer.name + "_layer", layer)

            self.log.store_architecture(self)
        else:
            self.data.build_architecture(self)

            for layer in self.all_layers:
                setattr(self, layer.name + "_layer", layer)

            self.log.store_architecture(self)

        # load weights learned by autoencoder
        self.load_weights(Cfg.xp_path + "/ae_pretrained_weights.p")

    def train(self, solver, n_epochs=10, save_at=0, save_to=''):

        self.solver = solver.lower()
        self.ae_solver = solver.lower()
        self.n_epochs = n_epochs
        self.save_at = save_at
        self.save_to = save_to

        self.log['solver'] = self.solver
        self.log['save_at'] = self.save_at

        self.compile_updates()

        from opt.sgd.train import train_network

        self.start_clock()
        train_network(self)
        self.stop_clock()

        # self.log.save_to_file()

    def evaluate(self, solver):

        # this could be simplified to only compiling the forwardpropagation...
        self.solver = solver.lower()  # needed for compiling backprop
        self.compile_updates()

        print("Evaluating network with current weights...")

        self.initialize_diagnostics(1)
        self.copy_parameters()

        # perform forward passes on training, val, and test set
        _, _ = performance(self, which_set='train', epoch=0, print_=True)
        _, _ = performance(self, which_set='val', epoch=0, print_=True)
        _, _ = performance(self, which_set='test', epoch=0, print_=True)

        print("Evaluation on train, val, and test set completed.")

    def log_results(self, filename=None):
        """
        log the results relevant for anomaly detection
        """

        self.ad_log['train_auc'] = self.diag['train']['auc'][-1]
        self.ad_log['train_accuracy'] = self.diag['train']['acc'][-1]
        self.ad_log['train_time'] = self.train_time

        self.ad_log['val_auc'] = self.diag['val']['auc'][-1]
        self.ad_log['val_accuracy'] = self.diag['val']['acc'][-1]

        self.ad_log['test_auc'] = self.diag['test']['auc'][-1]
        self.ad_log['test_accuracy'] = self.diag['test']['acc'][-1]
        self.ad_log['test_time'] = self.test_time

        self.ad_log.save_to_file(filename=filename)

    def addInputLayer(self, **kwargs):

        self.input_layer = InputLayer(name="input", **kwargs)
        self.input_layer.inp_ndim = len(kwargs["shape"])

    def addConvLayer(self, use_batch_norm=False, **kwargs):
        """
        Add convolutional layer.
        If batch norm flag is True, the convolutional layer
        will be followed by a batch-normalization layer
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_conv_layers += 1
        name = "conv%i" % self.n_conv_layers

        new_layer = ConvLayer(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)
        self.trainable_layers += (new_layer,)

        if use_batch_norm:
            self.n_bn_layers += 1
            name = "bn%i" % self.n_bn_layers
            self.all_layers += (BatchNorm(new_layer, name=name),)

    def addDenseLayer(self, use_batch_norm=False, **kwargs):
        """
        Add dense layer.
        If batch norm flag is True, the dense layer
        will be followed by a batch-normalization layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_dense_layers += 1
        name = "dense%i" % self.n_dense_layers

        new_layer = DenseLayer(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)
        self.trainable_layers += (new_layer,)

        if use_batch_norm:
            self.n_bn_layers += 1
            name = "bn%i" % self.n_bn_layers
            self.all_layers += (BatchNorm(new_layer, name=name),)

    def addSigmoidLayer(self, **kwargs):
        """
        Add sigmoid classification layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]
        new_layer = Sigmoid(input_layer, **kwargs)

        self.all_layers += (new_layer,)

        self.n_layers = len(self.all_layers)

    def addSoftmaxLayer(self, **kwargs):
        """
        Add softmax multi-class classification layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]
        new_layer = Softmax(input_layer, **kwargs)

        self.all_layers += (new_layer,)

        self.n_layers = len(self.all_layers)

    def addNormLayer(self, **kwargs):
        """
        Add layer which normalizes its input to length 1. 
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_norm_layers += 1
        name = "norm%i" % self.n_norm_layers

        new_layer = Norm(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addAbsLayer(self, **kwargs):
        """
        Add layer which returns the absolute value of its input.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_abs_layers += 1
        name = "abs%i" % self.n_abs_layers

        new_layer = Abs(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addReLU(self, **kwargs):
        """
        Add ReLU activation layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_relu_layers += 1
        name = "relu%i" % self.n_relu_layers

        new_layer = ReLU(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addLeakyReLU(self, **kwargs):
        """
        Add leaky ReLU activation layer. (with leakiness=0.01)
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_leaky_relu_layers += 1
        name = "leaky_relu%i" % self.n_leaky_relu_layers

        new_layer = LeakyReLU(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addMaxPool(self, **kwargs):
        """
        Add MaxPooling activation layer.
        """

        input_layer = self.input_layer if not self.all_layers\
            else self.all_layers[-1]

        self.n_maxpool_layers += 1
        name = "maxpool%i" % self.n_maxpool_layers

        new_layer = MaxPool(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addUpscale(self, **kwargs):
        """
        Add Upscaling activation layer.
        """

        input_layer = self.input_layer if not self.all_layers\
            else self.all_layers[-1]

        self.n_upscale_layers += 1
        name = "upscale%i" % self.n_upscale_layers

        new_layer = Upscale(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addDropoutLayer(self, **kwargs):
        """
        Add Dropout layer.
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_dropout_layers += 1
        name = "dropout%i" % self.n_dropout_layers

        new_layer = DropoutLayer(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addDimshuffleLayer(self, **kwargs):
        """
        Add Dimshuffle layer to reorder dimensions
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_dimshuffle_layers += 1
        name = "dimshuffle%i" % self.n_dimshuffle_layers

        new_layer = Dimshuffle(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def addReshapeLayer(self, **kwargs):
        """
        Add reshape layer to reshape dimensions
        """

        input_layer = self.input_layer if not self.all_layers \
            else self.all_layers[-1]

        self.n_reshape_layers += 1
        name = "reshape%i" % self.n_reshape_layers

        new_layer = Reshape(input_layer, name=name, **kwargs)

        self.all_layers += (new_layer,)

    def setFeatureLayer(self):
        """
        sets the currently highest layer of the current network to be the code layer (compression layer)
        """
        setattr(self, "feature_layer", self.all_layers[-1])

    def dump_weights(self, filename=None, pretrain=False):

        dump_weights(self, filename, pretrain=pretrain)

    def load_weights(self, filename=None):

        assert filename and os.path.exists(filename)

        load_weights(self, filename)

    def update_R(self):
        """
        method to update R while leaving the network parameters and center c fixed in a block coordinate optimization
        """

        print("Updating radius R...")

        # Get updates
        R = update_R(self.diag['train']['rep'], self.cvar.get_value(), solver=Cfg.R_update_solver,
                     scalar_method=Cfg.R_update_scalar_method, lp_obj=Cfg.R_update_lp_obj)

        # Update R
        self.Rvar.set_value(Cfg.floatX(R))

        print("Radius R updated.")

    def update_R_c(self):
        """
        method to update R and c while leaving the network parameters fixed in a block coordinate optimization
        """

        print("Updating radius R and center c...")

        # Get updates
        R, c = update_R_c(self.diag['train']['rep'], np.sum(self.diag['train']['rep'] ** 2, axis=1),
                          solver=Cfg.QP_solver)

        # Update values
        self.Rvar.set_value(Cfg.floatX(R))
        self.cvar.set_value(Cfg.floatX(c))

        print("Radius R and center c updated.")


    def initialize_diagnostics(self, n_epochs):
        """
        initialize diagnostics for the neural network
        """

        # data-dependent diagnostics for train, validation, and test set
        self.diag['train'] = NNetDataDiag(n_epochs=n_epochs, n_samples=self.data.n_train)
        self.diag['val'] = NNetDataDiag(n_epochs=n_epochs, n_samples=self.data.n_val)
        self.diag['test'] = NNetDataDiag(n_epochs=n_epochs, n_samples=self.data.n_test)

        # network parameter diagnostics
        self.diag['network'] = NNetParamDiag(self, n_epochs=n_epochs)

        # Best results (highest AUC on test set)
        self.auc_best = 0
        self.auc_best_epoch = 0  # determined by highest AUC on test set
        self.best_weight_dict = None


    def save_objective_and_accuracy(self, epoch, which_set, objective, accuracy):
        """
        save objective and accuracy of epoch
        """

        self.diag[which_set]['objective'][epoch] = objective
        self.diag[which_set]['acc'][epoch] = accuracy

    def save_initial_parameters(self):
        """
        save a copy of the initial network parameters for diagnostics.
        """

        self.W_init = []
        self.b_init = []

        for layer in self.trainable_layers:
            if layer.isdense | layer.isconv:
                self.W_init.append(None)
                self.b_init.append(None)

        i = 0
        for layer in self.trainable_layers:
            if layer.isdense | layer.isconv:
                self.W_init[i] = layer.W.get_value()
                if layer.b is not None:
                    self.b_init[i] = layer.b.get_value()
                i += 1

    def copy_parameters(self):
        """
        save a copy of the current network parameters in order to monitor the difference between epochs.
        """
        i = 0
        for layer in self.trainable_layers:
            if layer.isdense | layer.isconv:
                self.diag['network']['W_copy'][i] = layer.W.get_value()
                if layer.b is not None:
                    self.diag['network']['b_copy'][i] = layer.b.get_value()
                i += 1

    def copy_initial_parameters_to_cache(self):
        """
        Save a copy of the initial parameters in cache
        """
        self.diag['network']['W_copy'] = list(self.W_init)
        self.diag['network']['b_copy'] = list(self.b_init)

    def save_network_diagnostics(self, epoch, l2, R):
        """
        save diagnostics of the network
        """

        self.diag['network']['l2_penalty'][epoch] = l2
        self.log['l2_penalty'].append(float(l2))

        i = 0
        j = 0
        for layer in self.trainable_layers:
            if layer.isdense:
                self.diag['network']['W_norms'][i][:, epoch] = np.sum(layer.W.get_value() ** 2, axis=0)
                if layer.b is not None:
                    self.diag['network']['b_norms'][i][:, epoch] = layer.b.get_value() ** 2
                i += 1

            if layer.isdense | layer.isconv:
                dW = np.sqrt(np.sum((layer.W.get_value() - self.diag['network']['W_copy'][j]) ** 2))
                self.diag['network']['dW_norms'][j][epoch] = dW
                if layer.b is not None:
                    db = np.sqrt(np.sum((layer.b.get_value() - self.diag['network']['b_copy'][j]) ** 2))
                    self.diag['network']['db_norms'][j][epoch] = db
                j += 1

        # diagnostics only relevant for the SVDD loss
        if Cfg.svdd_loss:
            self.diag['network']['R'][epoch] = R
            self.diag['network']['c_norm'][epoch] = np.sqrt(np.sum(self.cvar.get_value() ** 2))

    def track_best_results(self, epoch):
        """
        Save network parameters where AUC on the test set was highest.
        """

        if self.diag['test']['auc'][epoch] > self.auc_best:
            self.auc_best = self.diag['test']['auc'][epoch]
            self.auc_best_epoch = epoch

            self.best_weight_dict = dict()

            for layer in self.trainable_layers:
                self.best_weight_dict[layer.name + "_w"] = layer.W.get_value()
                if layer.b is not None:
                    self.best_weight_dict[layer.name + "_b"] = layer.b.get_value()

            if Cfg.svdd_loss:
                self.best_weight_dict["R"] = self.Rvar.get_value()

    def dump_best_weights(self, filename):
        """
        pickle the network parameters, where AUC on the test set was highest.
        """

        with open(filename, 'wb') as f:
            pickle.dump(self.best_weight_dict, f)

        print("Parameters of best epoch saved in %s" % filename)

    def save_diagnostics(self, which_set, epoch, scores, rep_norm, rep, emp_loss, reconstruction_penalty):
        """
        save diagnostics for which_set of epoch
        """

        if self.data.n_classes == 2:

            if which_set == 'train':
                y = self.data._y_train
            if which_set == 'val':
                y = self.data._y_val
            if which_set == 'test':
                y = self.data._y_test

            self.diag[which_set]['scores'][:, epoch] = scores

            if sum(y) > 0:
                AUC = roc_auc_score(y, scores)
                self.diag[which_set]['auc'][epoch] = AUC
                self.log[which_set + '_auc'].append(float(AUC))
                print("{:32} {:.2f}%".format(which_set.title() + ' AUC:', 100. * AUC))

            scores_normal = scores[y == 0]
            scores_outlier = scores[y == 1]
            normal_summary = get_five_number_summary(scores_normal)
            outlier_summary = get_five_number_summary(scores_outlier)
            self.log[which_set + '_normal_scores_summary'].append(normal_summary)
            self.log[which_set + '_outlier_scores_summary'].append(outlier_summary)

            self.diag[which_set]['rep'] = rep
            self.diag[which_set]['rep_norm'][:, epoch] = rep_norm

            rep_norm_normal = rep_norm[y == 0]
            rep_norm_outlier = rep_norm[y == 1]
            normal_summary = get_five_number_summary(rep_norm_normal)
            outlier_summary = get_five_number_summary(rep_norm_outlier)
            self.log[which_set + '_normal_rep_norm_summary'].append(normal_summary)
            self.log[which_set + '_outlier_rep_norm_summary'].append(outlier_summary)

            if Cfg.svdd_loss:
                rep_mean = np.mean(rep, axis=0)
                self.diag[which_set]['output_mean_norm'][epoch] = np.sqrt(np.sum(rep_mean ** 2))
                self.diag[which_set]['c_mean_diff'][epoch] = np.sqrt(np.sum((rep_mean - self.cvar.get_value()) **2))

        self.diag[which_set]['reconstruction_penalty'][epoch] = reconstruction_penalty
        self.diag[which_set]['emp_loss'][epoch] = float(emp_loss)

        self.log[which_set + '_emp_loss'].append(float(emp_loss))


    def initialize_ae_diagnostics(self, n_epochs):
        """
        initialize diagnostic variables for autoencoder network.
        """

        self.train_time = 0
        self.test_time = 0
        self.best_weight_dict = None

        # data-dependent diagnostics for train, validation, and test set
        self.diag['train'] = NNetDataDiag(n_epochs=n_epochs, n_samples=self.data.n_train)
        self.diag['val'] = NNetDataDiag(n_epochs=n_epochs, n_samples=self.data.n_val)
        self.diag['test'] = NNetDataDiag(n_epochs=n_epochs, n_samples=self.data.n_test)

        # network parameters
        self.diag['network'] = {}
        self.diag['network']['l2_penalty'] = np.zeros(n_epochs, dtype=Cfg.floatX)

    def save_ae_diagnostics(self, which_set, epoch, error, scores, l2):
        """
        save autoencoder diagnostics for which_set
        """

        self.diag[which_set]['objective'][epoch] = error + l2
        self.diag[which_set]['emp_loss'][epoch] = error
        self.diag[which_set]['scores'][:, epoch] = scores

        self.diag['network']['l2_penalty'][epoch] = l2

        if self.data.n_classes == 2:

            if which_set == 'train':
                y = self.data._y_train
            if which_set == 'val':
                y = self.data._y_val
            if which_set == 'test':
                y = self.data._y_test

            if sum(y) > 0:
                AUC = roc_auc_score(y, scores)
                self.diag[which_set]['auc'][epoch] = AUC
