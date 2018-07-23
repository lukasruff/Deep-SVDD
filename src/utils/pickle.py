import cPickle as pickle
from config import Configuration as Cfg


def dump_weights(nnet, filename=None, pretrain=False):

    if filename is None:
        filename = nnet.pickle_filename

    weight_dict = dict()

    for layer in nnet.trainable_layers:
        weight_dict[layer.name + "_w"] = layer.W.get_value()
        if layer.b is not None:
            weight_dict[layer.name + "_b"] = layer.b.get_value()

    for layer in nnet.all_layers:
        if layer.isbatchnorm:
            weight_dict[layer.name + "_beta"] = layer.beta.get_value()
            weight_dict[layer.name + "_gamma"] = layer.gamma.get_value()
            weight_dict[layer.name + "_mean"] = layer.mean.get_value()
            weight_dict[layer.name + "_inv_std"] = layer.inv_std.get_value()

    if Cfg.svdd_loss and not pretrain:
            weight_dict["R"] = nnet.Rvar.get_value()

    with open(filename, 'wb') as f:
        pickle.dump(weight_dict, f)

    print("Parameters saved in %s" % filename)


def load_weights(nnet, filename=None):

    if filename is None:
        filename = nnet.pickle_filename

    with open(filename, 'rb') as f:
        weight_dict = pickle.load(f)

    for layer in nnet.trainable_layers:
        layer.W.set_value(weight_dict[layer.name + "_w"])
        if layer.b is not None:
            layer.b.set_value(weight_dict[layer.name + "_b"])

    for layer in nnet.all_layers:
        if layer.isbatchnorm:
            layer.beta.set_value(weight_dict[layer.name + "_beta"])
            layer.gamma.set_value(weight_dict[layer.name + "_gamma"])
            layer.mean.set_value(weight_dict[layer.name + "_mean"])
            layer.inv_std.set_value(weight_dict[layer.name + "_inv_std"])

    if Cfg.svdd_loss:
        if "R" in weight_dict:
            nnet.R_init = weight_dict["R"]

    print("Parameters loaded in network")


def dump_svm(model, filename=None):

    with open(filename, 'wb') as f:
        pickle.dump(model.svm, f)

    print("Model saved in %s" % filename)


def load_svm(model, filename=None):

    print("Loading model...")

    with open(filename, 'rb') as f:
        model.svm = pickle.load(f)

    print("Model loaded.")


def dump_kde(model, filename=None):

    with open(filename, 'wb') as f:
        pickle.dump(model.kde, f)

    print("Model saved in %s" % filename)


def load_kde(model, filename=None):

    print("Loading model...")

    with open(filename, 'rb') as f:
        model.kde = pickle.load(f)

    print("Model loaded.")

def dump_isoForest(model, filename=None):

    with open(filename, 'wb') as f:
        pickle.dump(model.isoForest, f)

    print("Model saved in %s" % filename)


def load_isoForest(model, filename=None):

    print("Loading model...")

    with open(filename, 'rb') as f:
        model.isoForest = pickle.load(f)

    print("Model loaded.")
