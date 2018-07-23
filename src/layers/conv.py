import lasagne.layers
import theano.tensor as T


class ConvLayer(lasagne.layers.Conv2DLayer):

    # for convenience
    isdense, isbatchnorm, isdropout, ismaxpool, isactivation = (False,) * 5
    isconv = True

    def __init__(self, incoming_layer, num_filters, filter_size, stride=(1, 1),
                 pad="valid", W=lasagne.init.GlorotUniform(gain='relu'),
                 b=lasagne.init.Constant(0.), flip_filters=True, name=None):

        lasagne.layers.Conv2DLayer.__init__(self, incoming_layer, num_filters,
                                            filter_size, name=name,
                                            stride=stride, pad=pad,
                                            untie_biases=False, W=W, b=b,
                                            nonlinearity=None,
                                            flip_filters=flip_filters,
                                            convolution=T.nnet.conv2d)

        self.inp_ndim = 4
        self.use_dc = False
