import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_five_number_summary(data, title, ylabel="Values", xlabel="Epochs", export_pdf=False, show=False):

    n_series = len(data)

    sns.set_style("white")
    sns.set_palette("colorblind", n_series)

    y_maxs = np.zeros(n_series)
    y_mins = np.zeros(n_series)
    epoch_maxs = np.zeros(n_series)

    i = 0
    for key in data:
        epochs = data[key].shape[1]
        x = np.arange(0, epochs)

        max = np.max(data[key], axis=0)
        upper_quant = np.percentile(data[key], 95, axis=0)
        median = np.median(data[key], axis=0)
        lower_quant = np.percentile(data[key], 5, axis=0)
        min = np.min(data[key], axis=0)

        plt.plot(x, median, '-', color=sns.color_palette()[i], label=key)
        plt.fill_between(x, lower_quant, upper_quant, alpha=0.25, facecolor=sns.color_palette()[i])
        plt.fill_between(x, min, max, alpha=0.25, facecolor=sns.color_palette()[i])

        y_maxs[i] = np.max(data[key])
        y_mins[i] = np.min(data[key])
        epoch_maxs[i] = epochs
        i += 1

    spread = np.max(y_maxs) - np.min(y_mins)
    plt.ylim(np.min(y_mins) - 0.025 * spread, np.max(y_maxs) + 0.025 * spread)
    plt.yscale('symlog')
    plt.xlim(0, np.max(epoch_maxs)-1)

    plt.grid(True, which="both")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if export_pdf:
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches=0.1)
        plt.clf()

    if show:
        plt.show()
