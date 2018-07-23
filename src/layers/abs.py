import lasagne.layers


class Abs(lasagne.layers.Layer):
    """
    This layer returns the absolute values of a dense layer
    """

    # for convenience
    isdense, isconv, isdropout, ismaxpool, isactivation, isbatchnorm = (False,) * 6

    def __init__(self, incoming_layer, **kwargs):
        super(Abs, self).__init__(incoming_layer, **kwargs)

        self.num_units = self.input_shape[1]

    def get_output_for(self, input, **kwargs):
        return abs(input)

    def get_output_shape_for(self, input_shape):
        return input_shape
