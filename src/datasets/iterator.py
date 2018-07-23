import numpy as np

from picklable_itertools import imap
from picklable_itertools.extras import partition_all

from fuel.schemes import BatchScheme

def iterate_batches(inputs, targets, batch_size, shuffle=False):

    n = len(targets)

    for (batch, idx) in indices_generator(shuffle=shuffle,
                                          batch_size=batch_size,
                                          n=n):
        yield inputs[batch], targets[batch], idx

def indices_generator(shuffle, batch_size, n):

    n_batches = int(np.ceil(n * 1. / batch_size))
    perm = np.arange(n_batches)

    if shuffle:
        np.random.shuffle(perm)

    for batch_idx in perm:
        start_idx = batch_idx * batch_size
        stop_idx = min(n, start_idx + batch_size)
        yield np.arange(start_idx, stop_idx), batch_idx


class MyScheme(BatchScheme):
    """
    Batch iterator. Inheriting and overriding fuel incomplete code.
    """

    def __init__(self, *args, **kwargs):

        self.rng = kwargs.pop('rng', None)

        if self.rng is None:
            self.rng = np.random.RandomState(0)

        super(MyScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):

        return imap(list, partition_all(self.batch_size, self.indices))
