import numpy as np

from collections import OrderedDict
from config import Configuration as Cfg
from utils.visualization.line_plot import plot_line
from utils.visualization.five_number_plot import plot_five_number_summary


def plot_diagnostics(nnet, xp_path, title_suffix, xlabel="Epochs", file_prefix=""):
    """
    a function to wrap different diagnostic plots
    """

    # plot train, validation, and test objective in one plot
    plot_objectives(nnet, xp_path, title_suffix, xlabel, file_prefix)

    # plot train, validation, and test objective (and their parts)
    plot_objective_with_parts(nnet, xp_path, title_suffix, xlabel, file_prefix)

    # plot accuracy
    plot_accuracy(nnet, xp_path, title_suffix, xlabel, file_prefix)

    # plot norms of network parameters (and parameter updates)
    plot_parameter_norms(nnet, xp_path, title_suffix, xlabel, file_prefix)

    if nnet.data.n_classes == 2:
        # plot auc
        plot_auc(nnet, xp_path, title_suffix, xlabel, file_prefix)

        # plot scores
        plot_scores(nnet, xp_path, title_suffix, xlabel, file_prefix)

        # plot norms of feature representations
        plot_representation_norms(nnet, xp_path, title_suffix, xlabel, file_prefix)

    if Cfg.svdd_loss:
        # plot diagnostics for center c in the network output space
        plot_center_c_diagnostics(nnet, xp_path, title_suffix, xlabel, file_prefix)

        # plot some random reconstructions if reconstruction regularizer is used
        if Cfg.reconstruction_penalty:
            plot_random_reconstructions(nnet, xp_path, title_suffix, file_prefix)


def plot_ae_diagnostics(nnet, xp_path, title_suffix):

    xlabel = "Epochs"
    file_prefix = "ae_"

    # plot train, validation, and test objective in one plot
    plot_objectives(nnet, xp_path, title_suffix, xlabel, file_prefix)

    # plot train, validation, and test objective
    plot_objective_with_parts(nnet, xp_path, title_suffix, xlabel, file_prefix, pretrain=True)

    # plot auc
    plot_auc(nnet, xp_path, title_suffix, xlabel, file_prefix)

    # plot scores
    plot_scores(nnet, xp_path, title_suffix, xlabel, file_prefix)


def plot_objectives(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot train, validation, and test objective in a combined plot
    """

    objectives = OrderedDict([("train", nnet.diag['train']['objective'])])
    if nnet.data.n_val > 0:
        objectives["val"] = nnet.diag['val']['objective']
    objectives["test"] = nnet.diag['test']['objective']

    plot_line(objectives, title="Objective " + title_suffix, xlabel=xlabel, ylabel="Objective", log_scale=True,
              export_pdf=(xp_path + "/" + file_prefix + "obj"))


def plot_objective_with_parts(nnet, xp_path, title_suffix, xlabel, file_prefix, pretrain=False):
    """
    plot train, validation, and test objective (and their parts)
    """

    for which_set in ['train', 'val', 'test']:

        if (which_set == 'val') & (nnet.data.n_val == 0):
            continue

        # Plot objective (and its parts)
        objective = OrderedDict([("objective", nnet.diag[which_set]['objective']),
                                 ("emp. loss", nnet.diag[which_set]['emp_loss']),
                                 ("l2 penalty", nnet.diag['network']['l2_penalty'])])
        if Cfg.reconstruction_penalty and not pretrain:
            objective["reconstruction penalty"] = nnet.diag[which_set]['reconstruction_penalty']
        if Cfg.svdd_loss and not pretrain:
            objective["R"] = nnet.diag['network']['R']

        title = which_set.title() + " objective " + title_suffix

        plot_line(objective, title=title, xlabel=xlabel, ylabel="Objective", log_scale=True,
                  export_pdf=(xp_path + "/" + file_prefix + "obj_" + which_set))


def plot_accuracy(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot accuracy of train, val, and test set per epoch.
    """

    acc = OrderedDict([("train", nnet.diag['train']['acc'])])
    if nnet.data.n_val > 0:
        acc["val"] = nnet.diag['val']['acc']
    acc["test"] = nnet.diag['test']['acc']

    plot_line(acc, title="Accuracy " + title_suffix, xlabel=xlabel, ylabel="Accuracy (%) ", y_min=-5, y_max=105,
              export_pdf=(xp_path + "/" + file_prefix + "accuracy"))


def plot_auc(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot auc time series of train, val, and test set.
    """

    auc = OrderedDict()

    for which_set in ['train', 'val', 'test']:

        if (which_set == 'val') & (nnet.data.n_val == 0):
            continue

        if which_set == 'train':
            y = nnet.data._y_train
        if which_set == 'val':
            y = nnet.data._y_val
        if which_set == 'test':
            y = nnet.data._y_test

        if sum(y) > 0:
            auc[which_set] = nnet.diag[which_set]['auc']

    plot_line(auc, title="AUC " + title_suffix, xlabel=xlabel, ylabel="AUC", y_min=-0.05, y_max=1.05,
              export_pdf=(xp_path + "/" + file_prefix + "auc"))


def plot_center_c_diagnostics(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot the norm of center c and the means of the outputs on the train, val, and test set per epoch
    """

    norms = OrderedDict([("c", nnet.diag['network']['c_norm']),
                         ("mean_train", nnet.diag['train']['output_mean_norm'])])
    if nnet.data.n_val > 0:
        norms["mean_val"] = nnet.diag['val']['output_mean_norm']
    norms["mean_test"] = nnet.diag['test']['output_mean_norm']

    plot_line(norms, title="Norms in output space " + title_suffix, xlabel=xlabel, ylabel="norm values ",
              log_scale=True, export_pdf=(xp_path + "/" + file_prefix + "c_mean_norms"))

    diffs = OrderedDict([("train", nnet.diag['train']['c_mean_diff'])])
    if nnet.data.n_val > 0:
        diffs["val"] = nnet.diag['val']['c_mean_diff']
    diffs["test"] = nnet.diag['test']['c_mean_diff']

    plot_line(diffs, title="Distance from c to means " + title_suffix, xlabel=xlabel, ylabel="distance ",
              log_scale=True, export_pdf=(xp_path + "/" + file_prefix + "c_mean_diffs"))


def plot_parameter_norms(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot norms of network parameters (and parameter updates)
    """

    # plot norms of parameters for each unit of dense layers
    params = OrderedDict()

    n_layer = 0
    for layer in nnet.trainable_layers:
        if layer.isdense:
            for unit in range(layer.num_units):
                name = "W" + str(n_layer + 1) + str(unit + 1)
                params[name] = nnet.diag['network']['W_norms'][n_layer][unit, :]
                if layer.b is not None:
                    name = "b" + str(n_layer + 1) + str(unit + 1)
                    params[name] = nnet.diag['network']['b_norms'][n_layer][unit, :]
            n_layer += 1

    plot_line(params, title="Norms of network parameters " + title_suffix, xlabel=xlabel, ylabel="Norm", log_scale=True,
              export_pdf=(xp_path + "/" + file_prefix + "param_norms"))

    # plot norms of parameter differences between updates for each layer
    params = OrderedDict()

    n_layer = 0
    for layer in nnet.trainable_layers:
        if layer.isdense | layer.isconv:
            name = "dW" + str(n_layer + 1)
            params[name] = nnet.diag['network']['dW_norms'][n_layer]
            if layer.b is not None:
                name = "db" + str(n_layer + 1)
                params[name] = nnet.diag['network']['db_norms'][n_layer]
            n_layer += 1

    plot_line(params, title="Absolute differences of parameter updates " + title_suffix, xlabel=xlabel, ylabel="Norm",
              log_scale=True, export_pdf=(xp_path + "/" + file_prefix + "param_diff_norms"))


def plot_scores(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot scores.
    """

    for which_set in ['train', 'val', 'test']:

        if (which_set == 'val') & (nnet.data.n_val == 0):
            continue

        if which_set == 'train':
            y = nnet.data._y_train
        if which_set == 'val':
            y = nnet.data._y_val
        if which_set == 'test':
            y = nnet.data._y_test

        # plot summary of scores
        scores = OrderedDict([('normal', nnet.diag[which_set]['scores'][y == 0])])
        if sum(y) > 0:
            scores['outlier'] = nnet.diag[which_set]['scores'][y == 1]

        title = "Summary of " + which_set + " scores " + title_suffix
        plot_five_number_summary(scores, title=title, xlabel=xlabel, ylabel="Score",
                                 export_pdf=(xp_path + "/" + file_prefix + "scores_" + which_set))


def plot_representation_norms(nnet, xp_path, title_suffix, xlabel, file_prefix):
    """
    plot norms of feature representations of train, val, and test set.
    """

    ylab = "Feature representation norm"

    for which_set in ['train', 'val', 'test']:

        if (which_set == 'val') & (nnet.data.n_val == 0):
            continue

        if which_set == 'train':
            y = nnet.data._y_train
        if which_set == 'val':
            y = nnet.data._y_val
        if which_set == 'test':
            y = nnet.data._y_test

        # plot summary of feature representation norms
        title = "Summary of " + which_set + " feature rep. norms " + title_suffix
        rep_norm = OrderedDict([('normal', nnet.diag[which_set]['rep_norm'][y == 0])])
        if sum(y) > 0:
            rep_norm['outlier'] = nnet.diag[which_set]['rep_norm'][y == 1]

        plot_five_number_summary(rep_norm, title=title, xlabel=xlabel, ylabel=ylab,
                                 export_pdf=(xp_path + "/" + file_prefix + "rep_norm_" + which_set))


def plot_random_reconstructions(nnet, xp_path, title_suffix, file_prefix, n_img=32):
    """
    plot the reconstructions of n_img randomly drawn images
    """

    # only plot reconstructions for image data
    if nnet.data._X_train.ndim != 4:
        return

    from utils.visualization.mosaic_plot import plot_mosaic

    random_idx = np.random.choice(nnet.data.n_train, n_img, replace=False)

    _, _, _, _, _, _, _, reconstruction, _, _ = nnet.forward(nnet.data._X_train[random_idx, ...],
                                                             nnet.data._y_train[random_idx])

    title = str(n_img) + " random autoencoder reconstructions " + title_suffix
    plot_mosaic(reconstruction, title=title, export_pdf=(xp_path + "/" + file_prefix + "ae_reconstructions"))
