import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_line(data, title, ylabel="Values", xlabel="Epochs", log_scale=False, y_min=None, y_max=None,
              export_pdf=False, show=False):

    sns_style = "white"
    sns_palette = "colorblind"
    sns.set(style=sns_style, palette=sns_palette)

    n_series = len(data)

    maxs = np.zeros(n_series)
    mins = np.zeros(n_series)

    i = 0
    for key in data:
        plt.plot(data[key], label=key)

        maxs[i] = np.max(data[key])
        mins[i] = np.min(data[key])
        i += 1

    spread = np.max(maxs) - np.min(mins)
    if y_min is not None:
        y1 = y_min
    else:
        y1 = np.min(mins) - 0.025 * spread
    if y_max is not None:
        y2 = y_max
    else:
        y2 = np.max(maxs) + 0.025 * spread
    plt.ylim(y1, y2)

    if log_scale:
        plt.yscale('symlog')
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
