import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_2Dscatter(data, title, export_pdf=False, show=False):

    sns_style = "white"
    sns_palette = "colorblind"
    sns.set(style=sns_style, palette=sns_palette)

    # n_series = len(data)

    # maxs = np.zeros(n_series)
    # mins = np.zeros(n_series)

    # i = 0
    for key in data:
        plt.scatter(data[key][:, 0], data[key][:, 1], marker='.', alpha=0.2, label=key)

        # maxs[i] = np.max(data[key])
        # mins[i] = np.min(data[key])
        # i += 1

    # spread = np.max(maxs) - np.min(mins)
    # plt.ylim(np.min(mins) - 0.025 * spread, np.max(maxs) + 0.025 * spread)

    plt.title(title)
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.legend()

    if export_pdf:
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches=0.1)
        plt.clf()

    if show:
        plt.show()
