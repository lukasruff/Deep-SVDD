import numpy as np

from config import Configuration as Cfg


def print_obj_and_acc(objective, accuracy, which_set):

    objective_str = '{} objective:'.format(which_set.title())
    accuracy_str = '{} accuracy:'.format(which_set.title())
    print("{:32} {:.5f}".format(objective_str, objective))
    print("{:32} {:.2f}%".format(accuracy_str, accuracy))


def performance(nnet, which_set, epoch=None, print_=False):

    floatX = Cfg.floatX

    objective = 0
    accuracy = 0
    batches = 0
    emp_loss = 0
    reconstruction_penalty = 0
    R = 0

    n = 0
    if which_set == 'train':
        n = nnet.data.n_train
    if which_set == 'val':
        n = nnet.data.n_val
    if which_set == 'test':
        n = nnet.data.n_test

    # prepare diagnostic variables
    scores = np.empty(n, dtype=floatX)
    if Cfg.svdd_loss:
        rep = np.empty((n, nnet.feature_layer.output_shape[1]), dtype=floatX)
    else:
        rep = np.empty((n, nnet.all_layers[-1].output_shape[1]), dtype=floatX)
    rep_norm = np.empty(n, dtype=floatX)


    for batch in nnet.data.get_epoch(which_set):
        inputs, targets, batch_idx = batch

        start_idx = batch_idx * Cfg.batch_size
        stop_idx = min(n, start_idx + Cfg.batch_size)

        if Cfg.softmax_loss:
            err, acc, b_scores, l2, b_loss = nnet.forward(inputs, targets)
            scores[start_idx:stop_idx] = b_scores[:, 0]
        elif Cfg.svdd_loss:

            err, acc, b_scores, l2, b_rec, b_rep, b_rep_norm, _, b_loss, R = nnet.forward(inputs, targets)

            scores[start_idx:stop_idx] = b_scores.flatten()
            rep[start_idx:stop_idx, :] = b_rep
            rep_norm[start_idx:stop_idx] = b_rep_norm
            reconstruction_penalty += b_rec
        else:
            err, acc, b_scores, l2, b_rep_norm, b_loss = nnet.forward(inputs, targets)
            scores[start_idx:stop_idx] = b_scores.flatten()
            rep_norm[start_idx:stop_idx] = b_rep_norm

        objective += err
        accuracy += acc
        emp_loss += b_loss
        batches += 1

    objective /= batches
    accuracy *= 100. / batches
    emp_loss /= batches
    reconstruction_penalty /= batches

    if print_:
        print_obj_and_acc(objective, accuracy, which_set)

    # save diagnostics

    nnet.save_objective_and_accuracy(epoch, which_set, objective, accuracy)
    nnet.save_diagnostics(which_set, epoch, scores, rep_norm, rep, emp_loss, reconstruction_penalty)

    # Save network parameter diagnostics (only once per epoch)
    if which_set == 'train':
        nnet.save_network_diagnostics(epoch, floatX(l2), floatX(R))

    # Track results of epoch with highest AUC on test set
    if which_set == 'test' and (nnet.data.n_classes == 2):
        nnet.track_best_results(epoch)

    return objective, accuracy


def ae_performance(nnet, which_set, epoch=None):

    n = 0
    if which_set == 'train':
        n = nnet.data.n_train
    if which_set == 'val':
        n = nnet.data.n_val
    if which_set == 'test':
        n = nnet.data.n_test

    l2 = 0
    batches = 0
    error = 0
    scores = np.empty(n)

    for batch in nnet.data.get_epoch(which_set):
        inputs, _, batch_idx = batch
        start_idx = batch_idx * Cfg.batch_size
        stop_idx = min(n, start_idx + Cfg.batch_size)

        err, l2, b_scores, _ = nnet.ae_forward(inputs)

        error += err * inputs.shape[0]
        scores[start_idx:stop_idx] = b_scores.flatten()
        batches += 1

    error /= n

    # # save diagnostics
    nnet.save_ae_diagnostics(which_set, epoch, error, scores, l2)

    return error
