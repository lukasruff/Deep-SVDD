import lasagne.layers


class DenseLayer(lasagne.layers.DenseLayer):

    # for convenience
    isconv, isbatchnorm, isdropout, ismaxpool, isactivation = (False,) * 5
    isdense = True

    def __init__(self, incoming_layer, num_units,
                 W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.),
                 name=None, nonlinearity=None, **kwargs):

        lasagne.layers.DenseLayer.__init__(self, incoming_layer, num_units,
                                           name=name, W=W, b=b,
                                           nonlinearity=nonlinearity, **kwargs)
        self.inp_ndim = 2
        self.use_dc = True
