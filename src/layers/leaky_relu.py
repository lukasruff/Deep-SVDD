import lasagne.layers


class LeakyReLU(lasagne.layers.NonlinearityLayer):

    # for convenience
    isdense, isbatchnorm, isconv, isdropout, ismaxpool, issigmoid = (False,) * 6
    isactivation, isrelu = (True,) * 2

    def __init__(self, incoming_layer, name=None,
                 nonlinearity=lasagne.nonlinearities.leaky_rectify, **kwargs):

        lasagne.layers.NonlinearityLayer.__init__(self, incoming_layer,
                                                  name=name,
                                                  nonlinearity=nonlinearity,
                                                  **kwargs)

        self.num_units = self.input_shape[1]
