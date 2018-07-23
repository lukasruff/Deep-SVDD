import lasagne.layers


class Softmax(lasagne.layers.NonlinearityLayer):

    # for convenience
    isdense, isbatchnorm, isconv, isdropout, ismaxpool, isrelu = (False,) * 6
    isactivation, issigmoid = (True,) * 2

    def __init__(self, incoming_layer, **kwargs):

        lasagne.layers.NonlinearityLayer.__init__(
            self, incoming_layer, name="softmax",
            nonlinearity=lasagne.nonlinearities.softmax, **kwargs)

        self.num_units = self.input_shape[1]
