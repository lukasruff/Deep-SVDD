import numpy as np



def weighted_degree_kernel(X1, X2, degree=1, weights=1):
    """
    Compute the weighted degree kernel matrix for one-hot encoded sequences.

    X1, X2: Tensor of training examples with shape 
        (n_examples, 1, len_dictionary, len_sequence)    
    degree: Degree of kernel
    weights: list or tuple with degree weights
    :return: Kernel matrix K
    """

    assert degree == len(weights)

    Klist = [None] * degree

    na = np.newaxis
    ones = np.ones(X2.shape)
    K = np.logical_and((X1[:, na, :, :, :] == X2[na, :, :, :, :]),
                       (X1[:, na, :, :, :] == ones[na, :, :, :, :]))
    K = K.sum(axis=(2, 3))

    # compute kernel matrix for each degree
    Klist[0] = K.sum(axis=2)
    for i in range(degree-1):
        K = (K[:, :, :-1] == K[:, :, 1:])
        Klist[i+1] = K.sum(axis=2)

    # compute weighted degree kernel
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(degree):
        K += weights[i] * Klist[i]

    return K
