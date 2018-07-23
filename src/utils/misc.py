import sys
import numpy as np


def flush_last_line(to_flush=1):

    for _ in range(to_flush):
        sys.stdout.write("\033[F")  # back to previous line
        sys.stdout.write("\033[K")  # clear line
        sys.stdout.flush()

def get_five_number_summary(data_array):
    """
    returns a list with the minimum, 5%-quantile, median, 95%-quantile, and
    maximum of data_array.
    """

    if data_array.size > 0:
        min = np.min(data_array)
        lower = np.percentile(data_array, 5)
        median = np.median(data_array)
        upper = np.percentile(data_array, 95)
        max = np.max(data_array)

        result = [min, lower, median, upper, max]
    else:
        result = [None]


    return result
