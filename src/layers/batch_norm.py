import lasagne.layers


class BatchNorm(lasagne.layers.BatchNormLayer):

    # for convenience
    isdense, isconv, isdropout, ismaxpool, isactivation = (False,) * 5
    isbatchnorm = True

    def __init__(self,
                 incoming_layer,
                 name=None,
                 **kwargs):

        lasagne.layers.BatchNormLayer.__init__(self, incoming_layer, name=name, **kwargs)
