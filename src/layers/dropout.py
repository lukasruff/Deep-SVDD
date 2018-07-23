import lasagne.layers


class DropoutLayer(lasagne.layers.DropoutLayer):

    # for convenience
    isdense, isbatchnorm, isconv, ismaxpool, isactivation = (False,) * 5
    isdropout = True

    def __init__(self, incoming_layer, name=None, p=0.5, rescale=True,
                 **kwargs):

        lasagne.layers.DropoutLayer.__init__(self, incoming_layer, name=name,
                                             p=p, rescale=rescale, **kwargs)

        self.num_units = self.input_shape[1]
