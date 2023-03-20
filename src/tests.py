#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import torch

import src.data_utils


def syntactic_check(g1, g2):
    graph1, graph2 = [nx.from_numpy_matrix(g[0].cpu().numpy(), create_using=nx.DiGraph) for g in [g1, g2]]
    assert (nx.is_tree(graph1) and nx.is_tree(graph2))
    same_node_n = [len(graph1.nodes()), len(graph2.nodes()), (len(graph1.nodes()) == len(graph2.nodes()))]
    type1, type2 = [np.nonzero(g[1].cpu().numpy())[1] for g in [g1, g2]]
    same_root = [type1[0], type2[0], (type1[0] == type2[0]) if len(type2) > 0 else False]
    leaf_n_1, leaf_n_2 = [sum([i >= 3 for i in t]) for t in [type1, type2]]
    same_leaf_n = [leaf_n_1, leaf_n_2, (leaf_n_1 == leaf_n_2)]
    tree_iso = nx.algorithms.isomorphism.tree_isomorphism(graph1.to_undirected(), graph2.to_undirected())
    unlabeled_iso = len(tree_iso) > 0
    simplified_iso = False
    labeled_iso = False
    if unlabeled_iso:
        nx.set_node_attributes(graph1, 0, name='type')
        nx.set_node_attributes(graph2, 0, name='type')
        for i in range(len(graph1.nodes())):
            graph1.nodes[i]['type'] = type1[i]
        for i in range(len(graph2.nodes())):
            graph2.nodes[i]['type'] = type2[i]
        iso_fn = nx.algorithms.isomorphism.categorical_node_match('type', 1)
        labeled_iso = nx.is_isomorphic(graph1, graph2, node_match=iso_fn)
        simplified_iso = labeled_iso
        if g1[1].shape[1] > 4:
            dict_1, dict_2 = [dict(), dict()]
            for i in range(len(graph1.nodes())):
                if graph1.nodes[i]['type'] > 3:
                    dict_1[i] = 3
            for j in range(len(graph2.nodes())):
                if graph1.nodes[j]['type'] > 3:
                    dict_2[j] = 3
            nx.set_node_attributes(graph1, dict_1, 'type')
            nx.set_node_attributes(graph2, dict_2, 'type')
            iso_fn = nx.algorithms.isomorphism.categorical_node_match('type', 0)
            simplified_iso = nx.is_isomorphic(graph1, graph2, node_match=iso_fn)
    return same_node_n, same_root, same_leaf_n, unlabeled_iso, simplified_iso, labeled_iso


def check_syntactic_validity(g, ariety=None):
    ariety = [2, 2, 1, 0] if ariety is None else ariety
    a, x = [g[0].cpu(), g[1].cpu()]
    correct = True
    for i in range(a.shape[0]):
        n_conn_i = len((a[i, :] == 1).nonzero().type(torch.LongTensor))
        type_i = (x[i, :] == 1).nonzero().item()
        ar_idx = type_i if type_i < len(ariety) else -1
        if ariety[ar_idx] != n_conn_i:
            correct = False
            break
    return correct


def get_results_row(adj, attr, acc, simplified=True):
    # create dataframe from list of [form, depth, n_nodes, n_and, n_or, n_not, n_leaves, n_dif_idx, root_type, acc]
    phi = src.data_utils.from_output_to_formula(adj, attr)
    res_list = [phi, phi.depth(), adj.shape[0], torch.sum(attr[:, 0]).item(), torch.sum(attr[:, 1]).item(),
                torch.sum(attr[:, 2]).item(), len(torch.where(~torch.any(adj, dim=1))[0])]
    if simplified is False:
        res_list.append(attr.shape[1] - 3)
    res_list.append(acc)
    return res_list
