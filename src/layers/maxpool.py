import lasagne.layers


class MaxPool(lasagne.layers.Pool2DLayer):

    # for convenience
    isdense, isbatchnorm, isconv, isdropout, isactivation = (False,) * 5
    ismaxpool = True

    def __init__(self, incoming_layer, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, name=None):

        lasagne.layers.Pool2DLayer.__init__(self, incoming_layer, pool_size,
                                            name=name, mode="max",
                                            stride=stride, pad=pad,
                                            ignore_border=ignore_border)

        self.inp_ndim = 4
        self.initialize_variables()

    def initialize_variables(self):

        self.pool_opts = {'ds': self.pool_size,
                          'st': self.stride,
                          'padding': self.pad,
                          'ignore_border': self.ignore_border}
