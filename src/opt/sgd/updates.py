import theano
import theano.tensor as T
import numpy as np
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates as l_updates
import lasagne.objectives as l_objectives

from lasagne.regularization import regularize_network_params, l2
from theano import shared
from config import Configuration as Cfg


def get_updates(nnet,
                train_obj,
                trainable_params,
                solver=None):

    implemented_solvers = ("sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam", "adamax")

    if solver not in implemented_solvers:
        nnet.sgd_solver = "adam"
    else:
        nnet.sgd_solver = solver

    if nnet.sgd_solver == "sgd":
        updates = l_updates.sgd(train_obj,
                                trainable_params,
                                learning_rate=Cfg.learning_rate)
    elif nnet.sgd_solver == "momentum":
        updates = l_updates.momentum(train_obj,
                                     trainable_params,
                                     learning_rate=Cfg.learning_rate,
                                     momentum=Cfg.momentum)
    elif nnet.sgd_solver == "nesterov":
        updates = l_updates.nesterov_momentum(train_obj,
                                              trainable_params,
                                              learning_rate=Cfg.learning_rate,
                                              momentum=Cfg.momentum)
    elif nnet.sgd_solver == "adagrad":
        updates = l_updates.adagrad(train_obj,
                                    trainable_params,
                                    learning_rate=Cfg.learning_rate)
    elif nnet.sgd_solver == "rmsprop":
        updates = l_updates.rmsprop(train_obj,
                                    trainable_params,
                                    learning_rate=Cfg.learning_rate,
                                    rho=Cfg.rho)
    elif nnet.sgd_solver == "adadelta":
        updates = l_updates.adadelta(train_obj,
                                     trainable_params,
                                     learning_rate=Cfg.learning_rate,
                                     rho=Cfg.rho)
    elif nnet.sgd_solver == "adam":
        updates = l_updates.adam(train_obj,
                                 trainable_params,
                                 learning_rate=Cfg.learning_rate)
    elif nnet.sgd_solver == "adamax":
        updates = l_updates.adamax(train_obj,
                                   trainable_params,
                                   learning_rate=Cfg.learning_rate)

    return updates


def get_l2_penalty(nnet, pow=2):
    """
    returns the l2 penalty on (trainable) network parameters combined as sum
    """

    l2_penalty = 0

    for layer in nnet.trainable_layers:
        l2_penalty = l2_penalty + T.sum(abs(layer.W) ** pow)

    return T.cast(l2_penalty, dtype='floatX')


def create_update(nnet):
    """
    create update for network given in argument
    """

    if nnet.data._X_train.ndim == 2:
        inputs = T.matrix('inputs')
    elif nnet.data._X_train.ndim == 4:
        inputs = T.tensor4('inputs')

    targets = T.ivector('targets')

    # compile theano functions
    if Cfg.softmax_loss:
        compile_update_softmax(nnet, inputs, targets)
    elif Cfg.svdd_loss:
        compile_update_svdd(nnet, inputs, targets)
    elif Cfg.reconstruction_loss:
        create_autoencoder(nnet)
    else:
        compile_update_default(nnet, inputs, targets)


def compile_update_default(nnet, inputs, targets):
    """
    create a SVM loss for network given in argument
    """

    floatX = Cfg.floatX
    C = Cfg.C

    if len(nnet.all_layers) > 1:
        feature_layer = nnet.all_layers[-2]
    else:
        feature_layer = nnet.input_layer
    final_layer = nnet.svm_layer
    trainable_params = lasagne.layers.get_all_params(final_layer, trainable=True)

    # Regularization
    if Cfg.weight_decay:
        l2_penalty = (floatX(0.5) / C) * get_l2_penalty(nnet)
    else:
        l2_penalty = T.cast(0, dtype='floatX')

    # Backpropagation
    prediction = lasagne.layers.get_output(final_layer, inputs=inputs, deterministic=False)
    objective, train_acc = final_layer.objective(prediction, targets)
    train_loss = T.cast((objective) / targets.shape[0], dtype='floatX')
    train_acc = T.cast(train_acc * 1. / targets.shape[0], dtype='floatX')
    train_obj = l2_penalty + train_loss
    updates = get_updates(nnet, train_obj, trainable_params, solver=nnet.solver)
    nnet.backprop = theano.function([inputs, targets], [train_obj, train_acc], updates=updates)

    # Forwardpropagation
    test_prediction = lasagne.layers.get_output(final_layer, inputs=inputs, deterministic=True)
    if nnet.data.n_classes == 2:
        scores = test_prediction[:, 1] - test_prediction[:, 0]
    else:
        scores = T.zeros_like(targets)
    objective, test_acc = final_layer.objective(test_prediction, targets)
    test_loss = T.cast(objective / targets.shape[0], dtype='floatX')
    test_acc = T.cast(test_acc * 1. / targets.shape[0], dtype='floatX')
    test_obj = l2_penalty + test_loss
    # get network feature representation
    test_rep = lasagne.layers.get_output(feature_layer, inputs=inputs, deterministic=True)
    test_rep_norm = test_rep.norm(L=2, axis=1)
    nnet.forward = theano.function([inputs, targets],
                                   [test_obj, test_acc, scores, l2_penalty, test_rep_norm, test_loss])


def compile_update_softmax(nnet, inputs, targets):
    """
    create a softmax loss for network given in argument
    """

    floatX = Cfg.floatX
    C = Cfg.C

    final_layer = nnet.all_layers[-1]
    trainable_params = lasagne.layers.get_all_params(final_layer, trainable=True)

    # Regularization
    if Cfg.weight_decay:
        l2_penalty = (floatX(0.5) / C) * get_l2_penalty(nnet)
    else:
        l2_penalty = T.cast(0, dtype='floatX')

    # Backpropagation
    prediction = lasagne.layers.get_output(final_layer, inputs=inputs, deterministic=False)

    if Cfg.ad_experiment:
        train_loss = T.mean(l_objectives.binary_crossentropy(prediction.flatten(), targets), dtype='floatX')
        train_acc = T.mean(l_objectives.binary_accuracy(prediction.flatten(), targets), dtype='floatX')
    else:
        train_loss = T.mean(l_objectives.categorical_crossentropy(prediction, targets), dtype='floatX')
        train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), targets), dtype='floatX')

    train_obj = T.cast(train_loss + l2_penalty, dtype='floatX')
    updates = get_updates(nnet, train_obj, trainable_params, solver=nnet.solver)
    nnet.backprop = theano.function([inputs, targets], [train_obj, train_acc], updates=updates)

    # Forwardpropagation
    test_prediction = lasagne.layers.get_output(final_layer, inputs=inputs, deterministic=True)

    if Cfg.ad_experiment:
        test_loss = T.mean(l_objectives.binary_crossentropy(test_prediction.flatten(), targets), dtype='floatX')
        test_acc = T.mean(l_objectives.binary_accuracy(test_prediction.flatten(), targets), dtype='floatX')
    else:
        test_loss = T.mean(l_objectives.categorical_crossentropy(test_prediction, targets), dtype='floatX')
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), targets), dtype='floatX')

    test_obj = T.cast(test_loss + l2_penalty, dtype='floatX')
    nnet.forward = theano.function([inputs, targets], [test_obj, test_acc, test_prediction, l2_penalty, test_loss])


def compile_update_svdd(nnet, inputs, targets):
    """
    create a Deep SVDD loss for network given in argument
    """

    floatX = Cfg.floatX

    ndim = nnet.data._X_train.ndim

    C = Cfg.C
    C_rec = Cfg.C_rec
    nu = Cfg.nu

    # initialize R
    if nnet.R_init > 0:
        nnet.Rvar = shared(floatX(nnet.R_init), name="R")
    else:
        nnet.Rvar = shared(floatX(1), name="R")  # initialization with R=1

    # Final Layer of the network
    final_layer = nnet.all_layers[-1]

    # SVDD Loss
    feature_layer = nnet.feature_layer
    rep = lasagne.layers.get_output(feature_layer, inputs=inputs, deterministic=False)

    # initialize c (0.5 in every feature representation dimension)
    rep_dim = feature_layer.num_units
    # nnet.cvar = shared(floatX(np.ones(rep_dim) * (1. / (rep_dim ** 0.5))),
    #                    name="c")
    nnet.cvar = shared(floatX(np.ones(rep_dim) * 0.5), name="c")

    dist = T.sum(((rep - nnet.cvar.dimshuffle('x', 0)) ** 2), axis=1, dtype='floatX')
    scores = dist - nnet.Rvar
    stack = T.stack([T.zeros_like(scores), scores], axis=1)
    loss = T.cast(T.sum(T.max(stack, axis=1)) / (inputs.shape[0] * nu), dtype='floatX')

    y_pred = T.argmax(stack, axis=1)
    acc = T.cast((T.sum(T.eq(y_pred.flatten(), targets), dtype='int32') * 1. / targets.shape[0]), 'floatX')

    # Network weight decay
    if Cfg.weight_decay:
        l2_penalty = (1/C) * get_l2_penalty(nnet)
    else:
        l2_penalty = T.cast(0, dtype='floatX')

    # Reconstruction regularization
    if Cfg.reconstruction_penalty:
        reconstruction = lasagne.layers.get_output(final_layer, inputs=inputs, deterministic=False)

        # use l2 or binary crossentropy loss (features are scaled to [0,1])
        if Cfg.ae_loss == "l2":
            rec_loss = lasagne.objectives.squared_error(reconstruction, inputs)
        if Cfg.ae_loss == "ce":
            rec_loss = lasagne.objectives.binary_crossentropy(reconstruction, inputs)

        rec_loss = T.sum(rec_loss, axis=range(1, ndim), dtype='floatX')
        rec_penalty = (1/C_rec) * T.mean(rec_loss)
    else:
        rec_penalty = T.cast(0, dtype='floatX')

    # Backpropagation (hard-margin: only minimizing everything to a ball centered at c)
    trainable_params = lasagne.layers.get_all_params(final_layer, trainable=True)
    if not Cfg.center_fixed:
        trainable_params.append(nnet.cvar)  # add center c to trainable parameters if it should not be fixed.

    avg_dist = T.mean(dist, dtype="floatX")

    obj_ball = T.cast(floatX(0.5) * (l2_penalty + rec_penalty) + avg_dist,
                      dtype='floatX')
    updates_ball = get_updates(nnet, obj_ball, trainable_params, solver=nnet.solver)
    nnet.backprop_ball = theano.function([inputs, targets], [obj_ball, acc], updates=updates_ball,
                                         on_unused_input='warn')

    # Backpropagation (without training R)
    obj = T.cast(floatX(0.5) * (l2_penalty + rec_penalty) + nnet.Rvar + loss,
                 dtype='floatX')
    updates = get_updates(nnet, obj, trainable_params, solver=nnet.solver)
    nnet.backprop_without_R = theano.function([inputs, targets], [obj, acc], updates=updates,
                                              on_unused_input='warn')

    # Backpropagation (with training R)
    trainable_params.append(nnet.Rvar)  # add radius R to trainable parameters
    updates = get_updates(nnet, obj, trainable_params, solver=nnet.solver)
    nnet.backprop = theano.function([inputs, targets], [obj, acc], updates=updates,
                                    on_unused_input='warn')


    # Forwardpropagation
    test_rep = lasagne.layers.get_output(feature_layer, inputs=inputs, deterministic=True)
    test_rep_norm = test_rep.norm(L=2, axis=1)

    test_dist = T.sum(((test_rep - nnet.cvar.dimshuffle('x', 0)) ** 2), axis=1, dtype='floatX')

    test_scores = test_dist - nnet.Rvar
    test_stack = T.stack([T.zeros_like(test_scores), test_scores], axis=1)
    test_loss = T.cast(T.sum(T.max(test_stack, axis=1)) / (inputs.shape[0]*nu), dtype='floatX')

    test_y_pred = T.argmax(test_stack, axis=1)
    test_acc = T.cast((T.sum(T.eq(test_y_pred.flatten(), targets), dtype='int32') * 1. / targets.shape[0]),
                      dtype='floatX')

    # Reconstruction regularization (with determinisitc=True)
    if Cfg.reconstruction_penalty:
        test_reconstruction = lasagne.layers.get_output(final_layer, inputs=inputs, deterministic=True)

        # use l2 or binary crossentropy loss (features are scaled to [0,1])
        if Cfg.ae_loss == "l2":
            test_rec_loss = lasagne.objectives.squared_error(test_reconstruction, inputs)
        if Cfg.ae_loss == "ce":
            test_rec_loss = lasagne.objectives.binary_crossentropy(test_reconstruction, inputs)

        test_rec_loss = T.sum(test_rec_loss, axis=range(1, ndim), dtype='floatX')
        test_rec_penalty = (1 / C_rec) * T.mean(test_rec_loss)
    else:
        test_reconstruction = lasagne.layers.get_output(final_layer, inputs=inputs, deterministic=True)
        test_rec_penalty = T.cast(0, dtype='floatX')

    test_obj = T.cast(floatX(0.5) * (l2_penalty + test_rec_penalty) + nnet.Rvar + test_loss, dtype='floatX')
    nnet.forward = theano.function([inputs, targets],
                                   [test_obj, test_acc, test_scores, floatX(0.5) * l2_penalty,
                                    floatX(0.5) * test_rec_penalty, test_rep,
                                    test_rep_norm, test_reconstruction, test_loss, nnet.Rvar],
                                   on_unused_input='warn')


def update_R_c(rep, rep_norm, solver='cvxopt', tol=1e-6):
    """
    Function to update R and c while leaving the network parameters fixed in a block coordinate optimization.
    Using quadratic programming of cvxopt.
    """

    assert solver in ('cvxopt', 'gurobi')

    n, d = rep.shape

    # Define QP
    P = (2 * np.dot(rep, rep.T)).astype(np.double)
    q = (-(rep_norm ** 2)).astype(np.double)
    G = (np.concatenate((np.eye(n), -np.eye(n)), axis=0)).astype(np.double)
    h = (np.concatenate(((1 / (Cfg.nu.get_value() * n)) * np.ones(n), np.zeros(n)), axis=0)).astype(np.double)
    A = (np.ones(n)).astype(np.double)
    b = 1

    if solver == 'cvxopt':

        from cvxopt import matrix
        from cvxopt.solvers import qp

        # Solve QP
        sol = qp(matrix(P),matrix(q),matrix(G),matrix(h),matrix(A).trans(),matrix(b, tc='d'))['x']
        a = np.array(sol).reshape(n)
        print("Sum of the elements of alpha: {:.3f}".format(np.sum(a)))

    if solver == 'gurobi':

        # Gurobi Python wrapper from https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/gurobi_.py

        from gurobipy import Model, QuadExpr, GRB, quicksum

        # setup model
        model = Model()
        x = {
            i: model.addVar(
                vtype=GRB.CONTINUOUS,
                name='x_%d' % i,
                lb=-GRB.INFINITY,
                ub=+GRB.INFINITY)
            for i in xrange(n)
        }
        model.update()

        # minimize 1/2 x.T * P * x + q * x
        obj = QuadExpr()
        rows, cols = P.nonzero()
        for i, j in zip(rows, cols):
            obj += 0.5 * x[i] * P[i, j] * x[j]
        for i in xrange(n):
            obj += q[i] * x[i]
        model.setObjective(obj, GRB.MINIMIZE)

        # subject to G * x <= h
        G_nonzero_rows = get_nonzero_rows(G)
        for i, row in G_nonzero_rows.iteritems():
            model.addConstr(quicksum(G[i, j] * x[j] for j in row) <= h[i])

        # subject to A * x == b
        A_nonzero_rows = get_nonzero_rows(A)
        for i, row in A_nonzero_rows.iteritems():
            model.addConstr(quicksum(A[i, j] * x[j] for j in row) == b[i])

        # Solve QP
        model.optimize()

        a = np.empty(n)
        for i in xrange(n):
            a[i] = model.getVarByName('x_%d' % i).x

    # Set new center c and radius R
    c = np.dot(a, rep).reshape(d).astype(np.float32)

    # Recover R (using the specified numeric tolerance on the range)
    n_svs = 0  # number of support vectors
    while n_svs == 0:
        lower = tol * (1/(Cfg.nu.get_value()*n))
        upper = (1-tol) * (1/(Cfg.nu.get_value()*n))
        idx_svs = (a > lower) & (a < upper)
        n_svs = np.sum(idx_svs)
        tol /= 10  # decrease tolerance if there are still no support vectors found

    print("Number of Support Vectors: {}".format(n_svs))
    R = np.mean(np.sum((rep[idx_svs] - c) ** 2, axis=1)).astype(np.float32)

    return R, c


def get_nonzero_rows(M):
    nonzero_rows = {}
    rows, cols = M.nonzero()
    for ij in zip(rows, cols):
        i, j = ij
        if i not in nonzero_rows:
            nonzero_rows[i] = []
        nonzero_rows[i].append(j)
    return nonzero_rows


def update_R(rep, center, solver='minimize_scalar', scalar_method='brent', lp_obj='primal', tol=0.001, **kwargs):
    """
    Function to update R while leaving the network parameters and center c fixed in a block coordinate optimization.
    Using scipy.optimize.minimize_scalar or linear programming of cvxopt.

    solver: should be either "minimize_scalar" (default) or "lp" (linear program)
    scalar_method: the optimization method used in minimize_scalar ('brent', 'bounded', or 'golden')
    lp_obj: should be either "primal" (default) or "dual"
    """

    assert solver in ("minimize_scalar", "lp")

    if solver == "minimize_scalar":

        from scipy.optimize import minimize_scalar

        assert scalar_method in ("brent", "bounded", "golden")

        print("Updating R with the {} method...".format(scalar_method))

        n, d = rep.shape
        dist = np.sum((rep - center) ** 2, axis=1, dtype=np.float32)

        # define deep SVDD objective function in R
        def f(x):
            return (x**2 + (1 / (Cfg.nu.get_value() * n)) *
                    np.sum(np.max(np.column_stack((np.zeros(n), dist - x**2)), axis=1), dtype=np.float32))

        # get lower and upper bounds around the (1-nu)-th quantile of distances
        bracket = None
        bounds = None

        upper_idx = int(np.max((np.floor(n * Cfg.nu.get_value() * 0.1), 1)))
        lower_idx = int(np.min((np.floor(n * Cfg.nu.get_value() * 1.1), n)))
        sort_idx = dist.argsort()
        upper = dist[sort_idx][-upper_idx]
        lower = dist[sort_idx][-lower_idx]

        if scalar_method in ("brent", "golden"):
            bracket = (lower, upper)

        elif scalar_method == "bounded":
            bounds = (lower, upper)

        # solve for R
        res = minimize_scalar(f, bracket=bracket, bounds=bounds, method=scalar_method)

        # Get new R
        R = res.x

    elif solver == "lp":

        from cvxopt import matrix
        from cvxopt.solvers import lp

        assert lp_obj in ("primal", "dual")

        print("Updating R on the {} objective...".format(lp_obj))

        n, d = rep.shape

        # Solve either primal or dual objective
        if lp_obj == "primal":

            # Define LP
            c = matrix(np.append(np.ones(1), (1 / (Cfg.nu.get_value() * n)) * np.ones(n), axis=0).astype(np.double))
            G = matrix(- np.concatenate((np.concatenate((np.ones(n).reshape(n,1), np.eye(n)), axis=1),
                                         np.concatenate((np.zeros(n).reshape(n, 1), np.eye(n)), axis=1)),
                                        axis=0).astype(np.double))
            h = matrix(np.append(- np.sum((rep - center) ** 2, axis=1), np.zeros(n), axis=0).astype(np.double))

            # Solve LP
            sol = lp(c, G, h)['x']

            # Get new R
            R = np.array(sol).reshape(n+1).astype(np.float32)[0]

        elif lp_obj == "dual":

            # Define LP
            c = matrix((np.sum((rep - center) ** 2, axis=1)).astype(np.double))
            G = matrix((np.concatenate((np.eye(n), -np.eye(n)), axis=0)).astype(np.double))
            h = matrix((np.concatenate(((1/(Cfg.nu.get_value()*n)) * np.ones(n), np.zeros(n)), axis=0)).astype(np.double))
            A = matrix((np.ones(n)).astype(np.double)).trans()
            b = matrix(1, tc='d')

            # Solve LP
            sol = lp(c, G, h, A, b)['x']
            a = np.array(sol).reshape(n)

            # Recover R (using the specified numeric tolerance on the range)
            n_svs = 0  # number of support vectors
            while n_svs == 0:
                lower = tol * (1/(Cfg.nu.get_value()*n))
                upper = (1-tol) * (1/(Cfg.nu.get_value()*n))
                idx_svs = (a > lower) & (a < upper)
                n_svs = np.sum(idx_svs)
                tol /= 10  # decrease tolerance if there are still no support vectors found

            R = np.median(np.array(c).reshape(n)[idx_svs]).astype(np.float32)

    return R


def create_autoencoder(nnet):
    """
    create autoencoder Theano update for network given in argument
    """

    floatX = Cfg.floatX
    C = Cfg.ae_C
    ndim = nnet.data._X_train.ndim

    if ndim == 2:
        inputs = T.matrix('inputs')
    elif ndim == 4:
        inputs = T.tensor4('inputs')

    final_layer = nnet.all_layers[-1]

    # Backpropagation
    trainable_params = lasagne.layers.get_all_params(final_layer, trainable=True)
    prediction = lasagne.layers.get_output(final_layer, inputs=inputs, deterministic=False)

    # use l2 or binary crossentropy loss (features are scaled to [0,1])
    if Cfg.ae_loss == "l2":
        loss = lasagne.objectives.squared_error(prediction, inputs)
    if Cfg.ae_loss == "ce":
        loss = lasagne.objectives.binary_crossentropy(prediction, inputs)

    scores = T.sum(loss, axis=range(1, ndim), dtype='floatX')
    loss = T.mean(scores)

    # Regularization
    if Cfg.ae_weight_decay:
        l2_penalty = (floatX(0.5) / C) * regularize_network_params(final_layer, l2)
    else:
        l2_penalty = T.cast(0, dtype='floatX')

    train_obj = loss + l2_penalty
    updates = get_updates(nnet, train_obj, trainable_params, solver=nnet.ae_solver)
    nnet.ae_backprop = theano.function([inputs], [loss, l2_penalty, scores], updates=updates)

    # Forwardpropagation
    test_prediction = lasagne.layers.get_output(final_layer, inputs=inputs, deterministic=True)

    # use l2 or binary crossentropy loss (features are scaled to [0,1])
    if Cfg.ae_loss == "l2":
        test_loss = lasagne.objectives.squared_error(test_prediction, inputs)
    if Cfg.ae_loss == "ce":
        test_loss = lasagne.objectives.binary_crossentropy(test_prediction, inputs)

    test_scores = T.sum(test_loss, axis=range(1, ndim), dtype='floatX')
    test_loss = T.mean(test_scores)

    nnet.ae_forward = theano.function([inputs],
                                      [test_loss, l2_penalty, test_scores, test_prediction])
