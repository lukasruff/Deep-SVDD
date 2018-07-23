import lasagne.layers
import theano.tensor as T


class Norm(lasagne.layers.Layer):
    """
    This layer normalizes the output of a dense layer such that it has l2-norm of 1.
    """

    # for convenience
    isdense, isconv, isdropout, ismaxpool, isactivation, isbatchnorm = (False,) * 6

    def __init__(self, incoming_layer, **kwargs):
        super(Norm, self).__init__(incoming_layer, **kwargs)

        self.num_units = self.input_shape[1]

    def get_output_for(self, input, **kwargs):
        norms = input.norm(L=2, axis=1).dimshuffle((0, 'x'))
        return input * T.tile(T.inv(norms), (1, self.input_shape[1]))

    def get_output_shape_for(self, input_shape):
        return input_shape
