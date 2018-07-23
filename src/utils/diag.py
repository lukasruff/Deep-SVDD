import numpy as np

from config import Configuration as Cfg


class NNetDataDiag(dict):
    """
    a class to capture data-dependent diagnostics (train, validation, or test set) of a neural network
    """

    def __init__(self, n_epochs, n_samples):

        dict.__init__(self)

        self['auc'] = np.zeros(n_epochs, dtype=Cfg.floatX)
        self['acc'] = np.zeros(n_epochs, dtype=Cfg.floatX)
        self['objective'] = np.zeros(n_epochs, dtype=Cfg.floatX)
        self['reconstruction_penalty'] = np.zeros(n_epochs, dtype=Cfg.floatX)
        self['emp_loss'] = np.zeros(n_epochs, dtype=Cfg.floatX)
        self['scores'] = np.zeros((n_samples, n_epochs), dtype=Cfg.floatX)
        self['rep_norm'] = np.zeros((n_samples, n_epochs), dtype=Cfg.floatX)

        # Deep SVDD specific diagnostics
        if Cfg.svdd_loss:
            self['output_mean_norm'] = np.zeros(n_epochs, dtype=Cfg.floatX)
            self['c_mean_diff'] = np.zeros(n_epochs, dtype=Cfg.floatX)


class NNetParamDiag(dict):
    """
    a class to capture diagnostics of the network parameters of a neural network
    """

    def __init__(self, nnet, n_epochs):

        dict.__init__(self)

        self['l2_penalty'] = np.zeros(n_epochs, dtype=Cfg.floatX)

        self['W_norms'] = []
        self['b_norms'] = []
        self['dW_norms'] = []
        self['db_norms'] = []
        self['W_copy'] = []
        self['b_copy'] = []

        for layer in nnet.trainable_layers:
            if layer.isdense:
                self['W_norms'].append(np.zeros((layer.num_units, n_epochs), dtype=Cfg.floatX))
                if layer.b is None:
                    self['b_norms'].append(None)
                else:
                    self['b_norms'].append(np.zeros((layer.num_units, n_epochs), dtype=Cfg.floatX))

            if layer.isdense | layer.isconv:
                self['dW_norms'].append(np.zeros(n_epochs, dtype=Cfg.floatX))
                if layer.b is None:
                    self['db_norms'].append(None)
                else:
                    self['db_norms'].append(np.zeros(n_epochs, dtype=Cfg.floatX))
                self['W_copy'].append(None)
                self['b_copy'].append(None)

        # Deep SVDD specific diagnostics
        if Cfg.svdd_loss:
            self['R'] = np.zeros(n_epochs, dtype=Cfg.floatX)
            self['c_norm'] = np.zeros(n_epochs, dtype=Cfg.floatX)
