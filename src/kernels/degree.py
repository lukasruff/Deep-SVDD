import numpy as np



def degree_kernel(X1, X2, degree=1):
    """
    Compute the degree kernel matrix for one-hot encoded sequences.

    X1, X2: Tensor of training examples with shape 
        (n_examples, 1, len_dictionary, len_sequence)    
    degree: Degree of kernel
    :return: Kernel matrix K
    """

    na = np.newaxis
    ones = np.ones(X2.shape)

    K = np.logical_and((X1[:, na, :, :, :] == X2[na, :, :, :, :]),
                       (X1[:, na, :, :, :] == ones[na, :, :, :, :]))

    K = K.sum(axis=(2, 3))

    for i in range(degree-1):
        K = (K[:, :, :-1] == K[:, :, 1:])

    K = K.sum(axis=2)

    return K