import time
import numpy as np

from config import Configuration as Cfg
from utils.monitoring import performance, ae_performance
from utils.visualization.diagnostics_plot import plot_diagnostics


def train_network(nnet):

    if Cfg.reconstruction_loss:
        nnet.ae_n_epochs = nnet.n_epochs
        train_autoencoder(nnet)
        return

    print("Starting training with %s" % nnet.sgd_solver)

    # save initial network parameters for diagnostics
    nnet.save_initial_parameters()
    if Cfg.nnet_diagnostics & Cfg.e1_diagnostics:
        # initialize diagnostics for first epoch (detailed diagnostics per batch)
        nnet.initialize_diagnostics(Cfg.n_batches + 1)
    else:
        nnet.initialize_diagnostics(nnet.n_epochs)

    # initialize c from mean of network feature representations in deep SVDD if specified
    if Cfg.svdd_loss and Cfg.c_mean_init:
        initialize_c_as_mean(nnet, Cfg.c_mean_init_n_batches)

    for epoch in range(nnet.n_epochs):

        # get copy of current network parameters to track differences between epochs
        nnet.copy_parameters()

        # In each epoch, we do a full pass over the training data:
        start_time = time.time()

        # learning rate decay
        if Cfg.lr_decay:
            decay_learning_rate(nnet, epoch)

        if Cfg.lr_drop and (epoch == Cfg.lr_drop_in_epoch):
            # Drop the learning rate in epoch specified in Cfg.lr_drop_after_epoch by factor Cfg.lr_drop_factor
            # Thus, a simple separation of learning into a "region search" and "finetuning" stage.
            lr_new = Cfg.floatX((1.0 / Cfg.lr_drop_factor) * Cfg.learning_rate.get_value())
            print("")
            print("Learning rate drop in epoch {} from {:.6f} to {:.6f}".format(
                epoch, Cfg.floatX(Cfg.learning_rate.get_value()), lr_new))
            print("")
            Cfg.learning_rate.set_value(lr_new)

        # train on epoch
        i_batch = 0
        for batch in nnet.data.get_epoch_train():

            if Cfg.nnet_diagnostics & Cfg.e1_diagnostics:
                # Evaluation before training
                if (epoch == 0) and (i_batch == 0):
                    _, _ = performance(nnet, which_set='train', epoch=i_batch)
                    if nnet.data.n_val > 0:
                        _, _ = performance(nnet, which_set='val', epoch=i_batch)
                    _, _ = performance(nnet, which_set='test', epoch=i_batch)

            # train
            inputs, targets, _ = batch

            if Cfg.svdd_loss:
                if Cfg.block_coordinate:
                    _, _ = nnet.backprop_without_R(inputs, targets)
                elif Cfg.hard_margin:
                    _, _ = nnet.backprop_ball(inputs, targets)
                else:
                    _, _ = nnet.backprop(inputs, targets)
            else:
                _, _ = nnet.backprop(inputs, targets)

            if Cfg.nnet_diagnostics & Cfg.e1_diagnostics:
                # Get detailed diagnostics (per batch) for the first epoch
                if epoch == 0:
                    _, _ = performance(nnet, which_set='train', epoch=i_batch+1)
                    if nnet.data.n_val > 0:
                        _, _ = performance(nnet, which_set='val', epoch=i_batch + 1)
                    _, _ = performance(nnet, which_set='test', epoch=i_batch+1)
                    nnet.copy_parameters()
                    i_batch += 1

        if (epoch == 0) & Cfg.nnet_diagnostics & Cfg.e1_diagnostics:
            # Plot diagnostics for first epoch
            plot_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix, xlabel="Batches", file_prefix="e1_")
            # Re-initialize diagnostics on epoch level
            nnet.initialize_diagnostics(nnet.n_epochs)
            nnet.copy_initial_parameters_to_cache()

        # Performance on training set (use forward pass with deterministic=True) to get the exact training objective
        train_objective, train_accuracy = performance(nnet, which_set='train', epoch=epoch, print_=True)

        # Adjust radius R for the SVDD hard-margin objective
        if Cfg.svdd_loss and (Cfg.hard_margin or (Cfg.block_coordinate and (epoch < Cfg.warm_up_n_epochs))):
            # set R to be the (1-nu)-th quantile of distances
            out_idx = int(np.floor(nnet.data.n_train * Cfg.nu.get_value()))
            sort_idx = nnet.diag['train']['scores'][:, epoch].argsort()
            R_new = nnet.diag['train']['scores'][sort_idx, epoch][-out_idx] + nnet.Rvar.get_value()
            nnet.Rvar.set_value(Cfg.floatX(R_new))

        # Update radius R and center c if block coordinate optimization is chosen
        if Cfg.block_coordinate and (epoch >= Cfg.warm_up_n_epochs) and ((epoch % Cfg.k_update_epochs) == 0):
            if Cfg.center_fixed:
                nnet.update_R()
            else:
                nnet.update_R_c()

        if Cfg.nnet_diagnostics:
            # Performance on validation and test set
            if nnet.data.n_val > 0:
                val_objective, val_accuracy = performance(nnet, which_set='val', epoch=epoch, print_=True)
            test_objective, test_accuracy = performance(nnet, which_set='test', epoch=epoch, print_=True)

            # log performance
            nnet.log['train_objective'].append(train_objective)
            nnet.log['train_accuracy'].append(train_accuracy)
            if nnet.data.n_val > 0:
                nnet.log['val_objective'].append(val_objective)
                nnet.log['val_accuracy'].append(val_accuracy)
            nnet.log['test_objective'].append(test_objective)
            nnet.log['test_accuracy'].append(test_accuracy)
            nnet.log['time_stamp'].append(time.time() - nnet.clock)

        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, nnet.n_epochs, time.time() - start_time))
        print('')

        # save model as required
        if epoch + 1 == nnet.save_at:
            nnet.dump_weights(nnet.save_to)

    # save train time
    nnet.train_time = time.time() - nnet.clock

    # Get final performance in last epoch if no running diagnostics are taken
    if not Cfg.nnet_diagnostics:

        nnet.initialize_diagnostics(1)
        nnet.copy_parameters()

        # perform forward passes on train, val, and test set
        print("Get final performance...")

        train_objective, train_accuracy = performance(nnet, which_set='train', epoch=0, print_=True)
        if nnet.data.n_val > 0:
            val_objective, val_accuracy = performance(nnet, which_set='val', epoch=0, print_=True)
        test_objective, test_accuracy = performance(nnet, which_set='test', epoch=0, print_=True)

        print("Evaluation completed.")

        # log performance
        nnet.log['train_objective'].append(train_objective)
        nnet.log['train_accuracy'].append(train_accuracy)
        if nnet.data.n_val > 0:
            nnet.log['val_objective'].append(val_objective)
            nnet.log['val_accuracy'].append(val_accuracy)
        nnet.log['test_objective'].append(test_objective)
        nnet.log['test_accuracy'].append(test_accuracy)
        nnet.log['time_stamp'].append(time.time() - nnet.clock)

    nnet.stop_clock()
    nnet.test_time = time.time() - (nnet.train_time + nnet.clock)

    # save final weights (and best weights in case of two-class dataset)
    nnet.dump_weights("{}/weights_final.p".format(Cfg.xp_path))
    if nnet.data.n_classes == 2:
        nnet.dump_best_weights("{}/weights_best_ep.p".format(Cfg.xp_path))


def decay_learning_rate(nnet, epoch):
    """
    decay the learning rate after epoch specified in Cfg.lr_decay_after_epoch
    """

    # only allow decay for non-adaptive solvers
    assert nnet.solver in ("sgd", "momentum", "adam")

    if epoch >= Cfg.lr_decay_after_epoch:
        lr_new = (Cfg.lr_decay_after_epoch / Cfg.floatX(epoch)) * nnet.learning_rate_init
        Cfg.learning_rate.set_value(Cfg.floatX(lr_new))
    else:
        return


def initialize_c_as_mean(nnet, n_batches, eps=0.1):
    """
    initialize c as the mean of the final layer representations from all samples propagated in n_batches
    """

    print("Initializing c...")

    # number of batches (and thereby samples) to initialize from
    if isinstance(n_batches, basestring) and n_batches == "all":
        n_batches = Cfg.n_batches
    elif n_batches > Cfg.n_batches:
        n_batches = Cfg.n_batches
    else:
        pass

    rep_list = list()

    i_batch = 0
    for batch in nnet.data.get_epoch_train():
        inputs, targets, _ = batch

        if i_batch == n_batches:
            break

        _, _, _, _, _, b_rep, _, _, _, _ = nnet.forward(inputs, targets)
        rep_list.append(b_rep)

        i_batch += 1

    reps = np.concatenate(rep_list, axis=0)
    c = np.mean(reps, axis=0)

    # If c_i is too close to 0 in dimension i, set to +-eps.
    # Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    nnet.cvar.set_value(c)

    # initialize R at the (1-nu)-th quantile of distances
    dist_init = np.sum((reps - c) ** 2, axis=1)
    out_idx = int(np.floor(len(reps) * Cfg.nu.get_value()))
    sort_idx = dist_init.argsort()
    nnet.Rvar.set_value(Cfg.floatX(dist_init[sort_idx][-out_idx]))

    print("c initialized.")


def train_autoencoder(nnet):

    if Cfg.ae_diagnostics:
        nnet.initialize_ae_diagnostics(nnet.ae_n_epochs)

    print("Starting training autoencoder with %s" % nnet.sgd_solver)

    for epoch in range(nnet.ae_n_epochs):

        start_time = time.time()

        if Cfg.ae_lr_drop and (epoch == Cfg.ae_lr_drop_in_epoch):
            # Drop the learning rate in epoch specified in Cfg.ae_lr_drop_after_epoch by factor Cfg.ae_lr_drop_factor
            # Thus, a simple separation of learning into a "region search" and "finetuning" stage.
            lr_new = Cfg.floatX((1.0 / Cfg.ae_lr_drop_factor) * Cfg.learning_rate.get_value())
            print("")
            print("Learning rate drop in epoch {} from {:.6f} to {:.6f}".format(
                epoch, Cfg.floatX(Cfg.learning_rate.get_value()), lr_new))
            print("")
            Cfg.learning_rate.set_value(lr_new)

        # In each epoch, we do a full pass over the training data:
        l2 = 0
        batches = 0
        train_err = 0
        train_scores = np.empty(nnet.data.n_train)

        for batch in nnet.data.get_epoch_train():
            inputs, _, batch_idx = batch
            start_idx = batch_idx * Cfg.batch_size
            stop_idx = min(nnet.data.n_train, start_idx + Cfg.batch_size)

            err, l2, b_scores = nnet.ae_backprop(inputs)

            train_err += err * inputs.shape[0]
            train_scores[start_idx:stop_idx] = b_scores.flatten()
            batches += 1

        train_err /= nnet.data.n_train

        # save train diagnostics and test performance on val and test data if specified
        if Cfg.ae_diagnostics:
            nnet.save_ae_diagnostics('train', epoch, train_err, train_scores, l2)

            # Performance on validation and test set
            if nnet.data.n_val > 0:
                val_err = ae_performance(nnet, which_set='val', epoch=epoch)
            test_err = ae_performance(nnet, which_set='test', epoch=epoch)


        # print results for epoch
        print("{:32} {:.5f}".format("Train error:", train_err))
        if Cfg.ae_diagnostics:
            if nnet.data.n_val > 0:
                print("{:32} {:.5f}".format("Val error:", val_err))
            print("{:32} {:.5f}".format("Test error:", test_err))
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, nnet.ae_n_epochs, time.time() - start_time))
        print("")

    # Get final performance in last epoch if no running diagnostics are taken
    if not Cfg.ae_diagnostics:
        nnet.initialize_ae_diagnostics(1)

        # perform forward passes on train, val, and test set
        print("Get final performance...")

        _ = ae_performance(nnet, which_set='train', epoch=0)
        if nnet.data.n_val > 0:
            _ = ae_performance(nnet, which_set='val', epoch=0)
        _ = ae_performance(nnet, which_set='test', epoch=0)

        print("Evaluation completed.")

    # save weights
    if Cfg.pretrain:
        nnet.dump_weights("{}/ae_pretrained_weights.p".format(Cfg.xp_path), pretrain=True)
    else:
        nnet.dump_weights("{}/weights_final.p".format(Cfg.xp_path))

    # if image data plot some random reconstructions
    if nnet.data._X_train.ndim == 4:
        from utils.visualization.mosaic_plot import plot_mosaic
        n_img = 32
        random_idx = np.random.choice(nnet.data.n_train, n_img, replace=False)
        _, _, _, reps = nnet.ae_forward(nnet.data._X_train[random_idx, ...])

        title = str(n_img) + " random autoencoder reconstructions"
        plot_mosaic(reps, title=title, export_pdf=(Cfg.xp_path + "/ae_reconstructions"))

    # plot diagnostics if specified
    if Cfg.ae_diagnostics & Cfg.pretrain:
        from utils.visualization.diagnostics_plot import plot_ae_diagnostics
        from utils.visualization.filters_plot import plot_filters

        # common suffix for plot titles
        str_lr = "lr = " + str(nnet.ae_learning_rate)
        C = int(Cfg.C.get_value())
        if not Cfg.weight_decay:
            C = None
        str_C = "C = " + str(C)
        title_suffix = "(" + nnet.ae_solver + ", " + str_C + ", " + str_lr + ")"

        # plot diagnostics
        plot_ae_diagnostics(nnet, Cfg.xp_path, title_suffix)

        # plot filters
        plot_filters(nnet, Cfg.xp_path, title_suffix, file_prefix="ae_", pretrain=True)
