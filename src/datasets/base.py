from iterator import indices_generator
from config import Configuration as Cfg


class DataLoader(object):

    def __init__(self, seed=0):

        # shuffling seed - important to have the same train / val
        # pseudo-random split when using pre-trained models
        self.seed = seed

        self.dataset_name = None

        self.n_train = None
        self.n_val = None

        self.n_classes = None

        self.data_path = None

        self.on_memory = None

    def check_base(self):

        for key in (self.__dict__):
            assert getattr(self, key) is not None, \
                "%s attribute should not be None" % key

        if self.on_memory:
            assert self.n_train == len(self._X_train)
            assert self.n_train == len(self._y_train)

            assert self.n_val == len(self._X_val)
            assert self.n_val == len(self._y_val)

            assert self.n_test == len(self._X_test)
            assert self.n_test == len(self._y_test)

    def check_specific(self):

        raise NotImplementedError("Should be replaced on each dataset")

    def check_all(self):

        self.check_base()
        self.check_specific()

    def build_architecture(self, nnet):

        raise NotImplementedError("Should be replaced on each dataset")

    def load_data(self):

        raise NotImplementedError("Should be replaced on each dataset")

    def get_epoch_train(self):

        assert self.on_memory, "only for data loaded on memory"

        for (batch, idx) in indices_generator(shuffle=True,
                                              batch_size=Cfg.batch_size,
                                              n=self.n_train):
            yield self._X_train[batch], self._y_train[batch], idx

    def get_epoch_val(self):

        for (batch, idx) in indices_generator(shuffle=False,
                                              batch_size=Cfg.batch_size,
                                              n=self.n_val):
            yield self._X_val[batch], self._y_val[batch], idx

    def get_epoch_test(self):

        for (batch, idx) in indices_generator(shuffle=False,
                                              batch_size=Cfg.batch_size,
                                              n=self.n_test):
            yield self._X_test[batch], self._y_test[batch], idx

    def get_epoch(self, which_set):

        assert which_set in ('train', 'val', 'test')

        if which_set == 'train':
            return self.get_epoch_train()

        if which_set == 'val':
            return self.get_epoch_val()

        if which_set == 'test':
            return self.get_epoch_test()
