from utils.visualization.mosaic_plot import plot_mosaic


def plot_outliers_and_most_normal(model, n, xp_path):
    """
    Plot outliers and most normal examples in ascending order of train, val, and test set.
    That is, the top left image is the most normal, the bottom right image the most anomalous example.
    """

    # reload images with original scale
    model.data.load_data(original_scale=True)

    for which_set in ['train', 'val', 'test']:

        if which_set == 'train':
            X = model.data._X_train
            n_samples = model.data.n_train
        if which_set == 'val':
            X = model.data._X_val
            n_samples = model.data.n_val
        if which_set == 'test':
            X = model.data._X_test
            n_samples = model.data.n_test

        if X.size > 0:  # only if set is specified

            idx_sort = model.diag[which_set]['scores'][:, -1].argsort()
            normals = X[idx_sort, ...][:n, ...]
            outliers = X[idx_sort, ...][-n:, ...]
            str_samples = "(" + str(n) + " of " + str(int(n_samples)) + ")"
            title_norm = which_set.title() + " set examples with ascending scores " + str_samples
            title_out = which_set.title() + " set outliers with ascending scores " + str_samples
            plot_mosaic(normals, title=title_norm, export_pdf=(xp_path + "/normals_" + which_set))
            plot_mosaic(outliers, title=title_out, export_pdf=(xp_path + "/outliers_" + which_set))

            # plot with scores at best epoch
            if model.best_weight_dict is not None:

                epoch = model.auc_best_epoch
                str_epoch = "at epoch " + str(epoch) + " "

                idx_sort = model.diag[which_set]['scores'][:, epoch].argsort()
                normals = X[idx_sort, ...][:n, ...]
                outliers = X[idx_sort, ...][-n:, ...]
                str_samples = "(" + str(n) + " of " + str(int(n_samples)) + ")"
                title_norm = (which_set.title() + " set examples with ascending scores " + str_epoch + str_samples)
                title_out = (which_set.title() + " set outliers with ascending scores " + str_epoch + str_samples)
                plot_mosaic(normals, title=title_norm, export_pdf=(xp_path + "/normals_" + which_set + "_best_ep"))
                plot_mosaic(outliers, title=title_out, export_pdf=(xp_path + "/outliers_" + which_set + "_best_ep"))

        else:
            pass
