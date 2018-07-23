import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_mosaic(W, title=None, canvas="white", export_pdf=False, show=False):
    """
    Plot filters of a convolutional layer or images in a mosaic. The function is capable of plotting grayscale or RGB
    images.

    :param W: W-parameter of a 2D convolutional layer with shape
        (num_filters, num_input_channels, filter_rows, filter_columns) already converted to a numpy-array or a set of
        images with shape (num_images, num_input_channels, image_rows, image_columns)
    """

    assert canvas in ("white", "black")

    W_shape = W.shape
    n_filters = W_shape[0]
    n_channels = W_shape[1]
    filter_shape = W_shape[2:]

    # rescale images to [0, 1]
    na = np.newaxis
    max_scale = W.max(axis=(2, 3))[..., na, na]
    min_scale = W.min(axis=(2, 3))[..., na, na]
    W = (W - min_scale) / (max_scale - min_scale)

    # make mosaic
    n_cols = int(np.floor(((2. / 1) * n_filters) ** 0.5))
    n_rows = (n_filters / n_cols) + int((n_filters % n_cols) > 0)

    if canvas == "white":
        mosaic = np.zeros((n_rows * filter_shape[0] + (n_rows - 1),
                           n_cols * filter_shape[1] + (n_cols - 1),
                           n_channels),
                          dtype=np.float32)
    if canvas == "black":
        mosaic = np.ones((n_rows * filter_shape[0] + (n_rows - 1),
                          n_cols * filter_shape[1] + (n_cols - 1),
                          n_channels),
                         dtype=np.float32)

    # account for grayscale images by removing single-dimensional entries
    if n_channels == 1:
        W = np.squeeze(W)
        mosaic = np.squeeze(mosaic)
    else:
        W = np.moveaxis(W, 1, -1)  # move channel-axis to last position


    paddedh = filter_shape[0] + 1
    paddedw = filter_shape[1] + 1

    for i in xrange(n_filters):
        row = int(np.floor(i / n_cols))
        col = i % n_cols

        idx_row_start = row * paddedh
        idx_row_end = row * paddedh + filter_shape[0]
        idx_col_start = col * paddedw
        idx_col_end = col * paddedw + filter_shape[1]

        mosaic[idx_row_start:idx_row_end, idx_col_start:idx_col_end] = W[i]

    plt.imshow(mosaic, cmap=cm.binary, interpolation='nearest', vmin=0, vmax=1)
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if title is not None:
        plt.title(title)

    if export_pdf:
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches=0.1)
        plt.clf()

    if show:
        plt.show()
