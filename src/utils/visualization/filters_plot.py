from utils.visualization.mosaic_plot import plot_mosaic


def plot_filters(nnet, xp_path, title_suffix, file_prefix="", pretrain=False):
    """
    Plot all filters of the first convolutional layer
    """

    assert hasattr(nnet, "conv1_layer")

    W = nnet.conv1_layer.W.get_value()
    title = "First layer filters " + title_suffix

    plot_mosaic(W, title=title, canvas="black", export_pdf=(xp_path + "/" + file_prefix + "filters"))

    if not pretrain and (nnet.best_weight_dict is not None):
        W_best = nnet.best_weight_dict["conv1_w"]
        title = ("First layer filters at epoch " + str(nnet.auc_best_epoch) + " " + title_suffix)
        plot_mosaic(W_best, title=title, canvas="black", export_pdf=(xp_path + "/" + file_prefix + "filters_best_ep"))
