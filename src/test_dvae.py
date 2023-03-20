#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from src.tests import syntactic_check, check_syntactic_validity, get_results_row


def test(model, arg, data_batches, data_loader=None, kind='test'):
    # test reconstruction accuracy
    model.eval()
    encode_times, decode_times = [arg.encode_times, arg.decode_times]
    nll = 0
    acc = []
    if kind == 'test':
        dataset_name = [i for i in os.listdir(arg.data) if i.startswith('test')][0]
        stat_name = 'stat_test' + dataset_name[9:] if (len(dataset_name) > 9 and dataset_name[:9] == 'test_data') \
            else 'stat_' + dataset_name
        if not os.path.isfile(arg.data + os.path.sep + stat_name):
            data_loader.generator.get_data_statistics(data_batches, arg.data + os.path.sep + stat_name)
    data_frame_list = [] if kind == 'test' else None
    validity, root, leaves, simp_iso, iso = [0 for _ in range(5)]
    for i, g in enumerate(data_batches):
        ground_truth = [data_batches[i][0][1:-1, 1:-1], data_batches[i][1][1:-1, 2:]]
        g = [g]
        mu, sigma = model.encode(g)
        _, ll, _ = model.loss(mu, sigma, g)
        nll += ll.item()
        validity_i, root_i, leaves_i, simp_iso_i, iso_i = [0 for _ in range(5)]
        for j in range(encode_times):
            z = model.reparameterize(mu, sigma)
            validity_j, root_j, leaves_j, simp_iso_j, iso_j = [0 for _ in range(5)]  # for single encoding
            for _ in range(decode_times):
                y = g[0][2].reshape(1, -1) if arg.conditional else None
                g_recon = model.decode(z, y=y)[0]  # assuming test is done for one formula at a time
                g_checks = [g_recon[0][1:-1, 1:-1], g_recon[1][1:-1, 2:]]
                valid = check_syntactic_validity(g_checks)
                validity_j += valid
                if valid:
                    same_node_n, same_root, same_leaf_n, unlabeled_iso, simp_iso, labeled_iso = \
                         syntactic_check(ground_truth, g_checks)
                    same_node_n_bool, same_root_bool, same_leaf_n_bool = \
                        [x[2] for x in [same_node_n, same_root, same_leaf_n]]
                    root_j, leaves_j, simp_iso_j, iso_j = [x + y for x, y in
                                               zip([root_j, leaves_j, simp_iso_j, iso_j],
                                                   [same_root_bool, same_leaf_n_bool, simp_iso, labeled_iso])]
            validity_i, root_i, leaves_i, simp_iso_i, iso_i = [x + y for x, y in
                                                               zip([validity_i, root_i, leaves_i, simp_iso_i, iso_i],
                                                                   [validity_j, root_j, leaves_j, simp_iso_j, iso_j])]
        # mean of single formula
        stats_i = np.divide(np.array([validity_i, root_i, leaves_i, simp_iso_i, iso_i]), encode_times*decode_times)
        file = open(arg.result_folder + os.path.sep + arg.model_name + os.path.sep + kind + "_results_single_"
                    + arg.model_name + ".txt", "a")
        file.write("Loss: %f N.nodes: %f, Acc. Metrics: %f %f %f %f %f\n" %
                   (ll.item(), ground_truth[0].shape[0], stats_i[0], stats_i[1], stats_i[2], stats_i[3], stats_i[4]))
        file.close()
        validity, root, leaves, simp_iso, iso = [x + y for x, y in zip([validity, root, leaves, simp_iso, iso],
                                                             [validity_i, root_i, leaves_i, simp_iso_i, iso_i])]
        if kind == 'test':
            data_frame_list.append(get_results_row(ground_truth[0], ground_truth[1], stats_i[4],
                                                   simplified=(not arg.var_indexes)))
    file = open(arg.result_folder + os.path.sep + arg.model_name + os.path.sep + kind + "_results_single_"
                + arg.model_name + ".txt", "a")
    file.write("\n")
    file.close()
    stats_test = np.divide(np.array([validity, root, leaves, simp_iso, iso]),
                           len(data_batches)*encode_times*decode_times)
    neg_ll = nll / len(data_batches)
    if kind == 'test':
        col_names = ['Formula', 'Depth', 'N-nodes', 'N-and', 'N-or', 'N-not', 'N-Leaves']
        if arg.var_indexes is True:
            col_names.append('N-leaves-idx')
        col_names.append('Accuracy')
        df = pd.DataFrame(data_frame_list, columns=col_names)
        df.to_csv(arg.result_folder + os.path.sep + arg.model_name + os.path.sep + "test_dataframe.csv")
    return neg_ll, stats_test
