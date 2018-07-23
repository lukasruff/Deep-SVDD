import lasagne.layers


class Upscale(lasagne.layers.Upscale2DLayer):
    """
    This layer performs unpooling over the last two dimensions of a 4D tensor.
    """

    # for convenience
    isdense, isbatchnorm, isconv, isdropout, isactivation, ismaxpool = (False,) * 6

    def __init__(self, incoming_layer, scale_factor, mode='repeat', name=None,
                 **kwargs):

        lasagne.layers.Upscale2DLayer.__init__(self, incoming_layer,
                                               scale_factor, mode=mode,
                                               name=name, **kwargs)

        self.inp_ndim = 4
