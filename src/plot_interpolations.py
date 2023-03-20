#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout

from data_utils import PropFormulaeDataset


def get_color_list(adj, c_list):
    types = np.nonzero(adj.cpu().numpy())[1]
    col = [c_list[t] for t in types]
    return col


def plot_interp(interp_list, phi_index, n_vars, fig_name, color_list):
    fig, ax = plt.subplots(1, len(interp_list), figsize=(2, 16))
    for j in range(len(interp_list[phi_index])):
        g = PropFormulaeDataset.build_tree(interp_list[phi_index][j])[0]
        x, a = PropFormulaeDataset.get_matrices(g, n_vars, simplified=False)
        graph_c_conv = get_color_list(a, color_list)
        pos_conv = graphviz_layout(g, prog="dot")
        nx.draw(g, pos_conv, node_color=graph_c_conv, with_labels=False, ax=ax[0, j])

    plt.tight_layout()
    plt.savefig(fig_name + '.pdf', transparent=True)
    plt.show()
