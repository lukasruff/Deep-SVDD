import lasagne.layers


class Dimshuffle(lasagne.layers.DimshuffleLayer):

    # for convenience
    isdense, isconv, isdropout, ismaxpool, isactivation, isbatchnorm = (False,) * 6

    def __init__(self, incoming_layer, name=None, **kwargs):

        lasagne.layers.DimshuffleLayer.__init__(self,
                                                incoming=incoming_layer,
                                                name=name,
                                                **kwargs)
