import lasagne.layers


class Sigmoid(lasagne.layers.NonlinearityLayer):

    # for convenience
    isdense, isbatchnorm, isconv, isdropout, ismaxpool, isrelu = (False,) * 6
    isactivation, issigmoid = (True,) * 2

    def __init__(self, incoming_layer, **kwargs):

        lasagne.layers.NonlinearityLayer.__init__(
            self, incoming_layer, name="sigmoid",
            nonlinearity=lasagne.nonlinearities.sigmoid, **kwargs)

        self.num_units = self.input_shape[1]
