import numpy as np
import theano


class Configuration(object):

    floatX = np.float32
    seed = 0

    # Final Layer
    softmax_loss = False
    svdd_loss = False
    reconstruction_loss = False

    # Optimization
    batch_size = 200
    learning_rate = theano.shared(floatX(1e-4), name="learning rate")
    lr_decay = False
    lr_decay_after_epoch = 10
    lr_drop = False  # separate into "region search" and "fine-tuning" stages
    lr_drop_factor = 10
    lr_drop_in_epoch = 50
    momentum = theano.shared(floatX(0.9), name="momentum")
    rho = theano.shared(floatX(0.9), name="rho")
    use_batch_norm = False  # apply batch normalization

    eps = floatX(1e-8)

    # Network architecture
    leaky_relu = False
    dropout = False
    dropout_architecture = False

    # Pre-training and autoencoder configuration
    weight_dict_init = False
    pretrain = False
    ae_loss = "l2"
    ae_lr_drop = False  # separate into "region search" and "fine-tuning" stages
    ae_lr_drop_factor = 10
    ae_lr_drop_in_epoch = 50
    ae_weight_decay = True
    ae_C = theano.shared(floatX(1e3), name="ae_C")

    # Regularization
    weight_decay = True
    C = theano.shared(floatX(1e3), name="C")
    reconstruction_penalty = False
    C_rec = theano.shared(floatX(1e3), name="C_rec")  # Hyperparameter of the reconstruction penalty

    # SVDD
    nu = theano.shared(floatX(.2), name="nu")
    c_mean_init = False
    c_mean_init_n_batches = "all"
    hard_margin = False
    block_coordinate = False
    k_update_epochs = 5  # update R and c only every k epochs, i.e. always train the network for k epochs in one block.
    R_update_solver = "minimize_scalar"  # "minimize_scalar" (default) or "lp" (linear program)
    R_update_scalar_method = "bounded"  # optimization method used in minimize_scalar ('brent', 'bounded', or 'golden')
    R_update_lp_obj = "primal" # on which objective ("primal" or "dual") should R be optimized if LP?
    center_fixed = True  # determine if center c should be fixed or not (in which case c is an optimization parameter)
    QP_solver = 'cvxopt'  # the library to use for solving the QP (or LP). One of ("cvxopt" or "gurobi")
    warm_up_n_epochs = 10  # iterations until R and c are also getting optimized

    # Data preprocessing
    out_frac = floatX(.1)
    ad_experiment = False
    pca = False
    unit_norm_used = "l2"  # "l2" or "l1"
    gcn = False
    zca_whitening = False

    # MNIST dataset parameters
    mnist_val_frac = 1./6
    mnist_bias = True
    mnist_rep_dim = 32
    mnist_architecture = 1  # choose one of the implemented architectures
    mnist_normal = 0
    mnist_outlier = -1

    # CIFAR-10 dataset parameters
    cifar10_val_frac = 1./5
    cifar10_bias = True
    cifar10_rep_dim = 128
    cifar10_architecture = 1  # choose one of the implemented architectures
    cifar10_normal = 1
    cifar10_outlier = -1

    # GTSRB dataset parameters
    gtsrb_rep_dim = 32

    # Plot parameters
    xp_path = "../log/"
    title_suffix = ""

    # SVM parameters
    svm_C = floatX(1.0)
    svm_nu = floatX(0.2)
    svm_GridSearchCV = False

    # KDE parameters
    kde_GridSearchCV = False

    # Diagnostics (should diagnostics be retrieved? Training is faster without)
    nnet_diagnostics = True  # diagnostics for neural networks in general (including Deep SVDD)
    e1_diagnostics = True  # diagnostics for neural networks in first epoch
    ae_diagnostics = True  # diagnostics for autoencoders
