#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import networkx as nx

from src.gat_layer import GAT


# implements decoder, loss and GRU encoder
class LogicVAE(nn.Module):
    def __init__(self, v_types, arity=None, max_v=None, start_idx=0, end_idx=1, h_dim=200, z_dim=56,
                 bidirectional=False, gat=False, v_id=False, conditional=False, semantic_length=100, device=None):
        super(LogicVAE, self).__init__()
        # number of different nde types
        self.v_types = v_types
        # number of children each node type can have (including start and end type)
        self.type_arity = [1, 0, 2, 2, 1, 0] if arity is None else arity
        # max number of nodes a generated graph can have
        self.max_n_vert = max_v
        # type index of synthetic start and end nodes
        self.start_idx = start_idx
        self.end_idx = end_idx
        # whether to use GAT as encoder
        self.attentional = gat
        # hidden dimension
        self.hidden_size = h_dim
        # latent dimension
        self.latent_size = z_dim
        # whether vertex should be given a unique identifier (and corresponding node embedding dimension)
        self.v_id = v_id
        self.vertex_state_size = self.hidden_size + max_v if v_id else self.hidden_size
        # whether to use bidirectional message passing
        self.bidirectional = bidirectional
        # number of convolutional layers (if conv-style message passing is used)
        self.n_layers = None
        # whether to implement CVAE (or vanilla VAE)
        self.conditional = conditional
        self.encoding_size = h_dim if not self.conditional else h_dim + semantic_length
        if self.conditional:
            # dimension of the semantic context vector
            assert (semantic_length > 0)
            self.semantic_length = semantic_length
        # whether to run on cpu or gpu
        self.device = device
        # ENCODER
        # input to GRU is: one-hot encoding of current node type - combined incoming messages to current node
        self.gru_enc_forward = nn.GRUCell(self.v_types, self.hidden_size)
        # gated sum utilities
        self.gate_forward = nn.Sequential(nn.Linear(self.vertex_state_size, self.hidden_size), nn.Sigmoid())
        self.map_forward = nn.Sequential(nn.Linear(self.vertex_state_size, self.hidden_size, bias=False), )
        # latent space MLP
        self.mlp_mean = nn.Linear(self.encoding_size, self.latent_size)
        self.mlp_std = nn.Linear(self.encoding_size, self.latent_size)
        if self.bidirectional:
            self.gru_enc_backward = nn.GRUCell(self.v_types, self.hidden_size)
            self.gate_backward = nn.Sequential(nn.Linear(self.vertex_state_size, self.hidden_size), nn.Sigmoid())
            self.map_backward = nn.Sequential(nn.Linear(self.vertex_state_size, self.hidden_size, bias=False), )
            self.unify_hidden_graph = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), )

        # DECODER
        self.gru_dec = nn.GRUCell(self.v_types, self.hidden_size)
        from_latent_input_dim = self.latent_size if not self.conditional else self.latent_size + self.semantic_length
        self.from_latent = nn.Linear(from_latent_input_dim, self.hidden_size)
        self.vertex_type = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 4), nn.ReLU(),
                                         nn.Linear(self.hidden_size * 4, self.v_types-2))
        self.tanh = nn.Tanh()
        self.logsoftmax = nn.LogSoftmax(1)
        self.negLogLikelihood = nn.NLLLoss(reduction='sum')

    @property
    def _max_n_vert(self):
        return self.max_n_vert

    @_max_n_vert.setter
    def _max_n_vert(self, max_v):
        self.max_n_vert = max_v

    def message_to(self, g_in, v_idx, prop_net, hidden_single, hidden_agg=None, reverse=False):
        # send all messages to vertex v from predecessors
        # g_in is a list of graphs (i.e. of list [adj, attr])
        n_vert = [g[0].shape[0] for g in g_in]
        graphs_idx = [i for i, _ in enumerate(g_in) if n_vert[i] > v_idx]
        graphs = [g for i, g in enumerate(g_in) if i in graphs_idx]
        if len(graphs) == 0:
            return None, hidden_single
        # extract adjacency and feature matrix for each graph in the list
        adj = [g[0] for g in graphs]
        type_v = [g[1][v_idx, :] for g in graphs]
        # zero-padding for consistent dimensionality
        type_v = [F.pad(t, pad=(0, self.v_types - len(t)), mode='constant', value=0) if self.v_types - len(t) > 0 else t
                  for t in type_v]
        x = torch.cat(type_v, 0).reshape(len(type_v), -1)
        v_ids = None
        if hidden_agg is not None:
            # original size: [#graphs, hidden_dim]
            hidden_agg = hidden_agg[graphs_idx]
        if reverse:
            # find successor of current node for each graph in the list
            succ_idx = [(a[v_idx, :] == 1).nonzero().type(torch.LongTensor) for a in adj]
            # h_pred has size [#graphs, #succ, hidden_dim]
            h_pred = [[hidden_single[g_idx][s_idx, :] for s_idx in succ_idx[i]] for i, g_idx in enumerate(graphs_idx)]
            if self.v_id:
                succs = [succ.unsqueeze(0).t() if len(succ.shape) < 2 else succ.t() for succ in succ_idx]
                v_ids = [torch.zeros((len(h_pred[i]), self.max_n_vert)).scatter_(1, succ, 1).to(self.device)
                         for i, succ in enumerate(succs)]
            # gated sum of messages
            gate, mapper = self.gate_backward, self.map_backward
        else:
            # find predecessor of current node for each graph in the list
            pred_idx = [(a[:, v_idx] == 1).nonzero().type(torch.LongTensor) for a in adj]
            # h_pred has size [#graphs, #pred, hidden_dim]
            h_pred = [[hidden_single[g_idx][p_idx, :] for p_idx in pred_idx[i]] for i, g_idx in enumerate(graphs_idx)]
            if self.v_id:
                preds = [pred.unsqueeze(0).t() if len(pred.shape) < 2 else pred.t() for pred in pred_idx]
                v_ids = [torch.zeros((len(h_pred[i]), self.max_n_vert)).scatter_(1, pred, 1).to(self.device)
                         for i, pred in enumerate(preds)]
            gate, mapper = self.gate_forward, self.map_forward
        if self.v_id:
            h_pred = [[torch.cat([x[i], y[i:i + 1]], 1) for i in range(len(x))] for x, y in zip(h_pred, v_ids)]
        if hidden_agg is None:
            max_pred = max([len(p) for p in h_pred])
            if max_pred == 0:
                hidden_agg = torch.zeros(len(graphs), self.hidden_size).to(self.device)
            else:
                h_pred = [torch.cat(h_p + [torch.zeros(max_pred - len(h_p), self.vertex_state_size).to(self.device)],
                                    0).unsqueeze(0) for h_p in h_pred]
                h_pred = torch.cat(h_pred, 0)
                hidden_agg = (gate(h_pred) * mapper(h_pred)).sum(1)
        new_hidden_v = prop_net(x.to(self.device), hidden_agg.to(self.device))
        for i, g_idx in enumerate(graphs_idx):
            hidden_single[g_idx][v_idx, :] = new_hidden_v[i:i + 1]
        return new_hidden_v, hidden_single

    def message_from(self, g_in, v_idx, prop_net, hidden_single, hidden_agg_zero=None, reverse=False):
        topo_order = range(v_idx, -1, -1) if reverse else range(v_idx, self.max_n_vert)
        # send message to v and update v state
        hidden_v, hidden_single = self.message_to(g_in, v_idx, prop_net, hidden_single, hidden_agg=hidden_agg_zero,
                                                  reverse=reverse)
        # send message to topologically following nodes
        for v_next in topo_order[1:]:
            _, hidden_single = self.message_to(g_in, v_next, prop_net, hidden_single, reverse=reverse)
        return hidden_single

    def get_graph_state(self, g_in, hidden_single_forward, hidden_single_backward=None, intermediate=False,
                        decode=False):
        semantic_vects = []
        hidden_graph = []
        for g, graph in enumerate(g_in):
            # take hidden state of last node (e.g. end or last added)
            hidden_g = hidden_single_forward[g][-1, :]
            if self.bidirectional and not decode:
                assert (hidden_single_backward is not None)
                hidden_g_back = hidden_single_backward[g][0, :]
                hidden_g = torch.cat([hidden_g, hidden_g_back], 0)
            hidden_graph.append(hidden_g.unsqueeze(0))
            if self.conditional and not intermediate:
                semantic_vects.append(graph[2].reshape(1, -1))
        hidden_graph = torch.cat(hidden_graph, 0)
        if self.bidirectional and not decode:
            hidden_graph = self.unify_hidden_graph(hidden_graph)
        hidden_graph = hidden_graph.reshape(-1, self.hidden_size)
        # during decoding or loss computation we don't use the semantic information
        if self.conditional and not intermediate:
            semantic_vects = torch.cat(semantic_vects, 0)
            hidden_graph = torch.cat([hidden_graph, semantic_vects], 1)
        return hidden_graph

    def encode(self, g_in):
        if type(g_in) is not list:
            g_in = [g_in]
        n_vars = [g[0].shape[0] for g in g_in]
        hidden_single_forward = [torch.zeros(n, self.hidden_size).to(self.device) for i, n in enumerate(n_vars)]
        hidden_agg_init = torch.zeros(len(g_in), self.hidden_size).to(self.device)
        # forward pass
        hidden_single_forward = self.message_from(g_in, 0, self.gru_enc_forward, hidden_single=hidden_single_forward,
                                                  hidden_agg_zero=hidden_agg_init, reverse=False)
        # backward pass
        if self.bidirectional:
            hidden_single_backward = [torch.zeros(n, self.hidden_size).to(self.device)
                                      for i, n in enumerate(n_vars)]
            hidden_single_backward = self.message_from(g_in, self.max_n_vert - 1, self.gru_enc_backward,
                                                       hidden_single=hidden_single_backward,
                                                       hidden_agg_zero=hidden_agg_init, reverse=True)
            hidden_g = self.get_graph_state(g_in, hidden_single_forward, hidden_single_backward, decode=False,
                                            intermediate=False)
            for hs in hidden_single_backward:
                del hs  # to save GPU memory
        else:
            hidden_g = self.get_graph_state(g_in, hidden_single_forward, decode=False, intermediate=False)
        mu, sigma = self.mlp_mean(hidden_g), self.mlp_std(hidden_g)
        for hs in hidden_single_forward:
            del hs  # to save GPU memory
        return mu, sigma

    def reparameterize(self, mu, sigma, eps=0.01):
        if self.training:
            return (torch.randn_like(sigma) * eps).mul(sigma.mul(0.5).exp_()).add_(mu)
        else:
            return mu

    def add_node_adj(self, adj):
        new_adj = torch.zeros(adj.shape[0] + 1, adj.shape[1] + 1).to(self.device)
        new_adj[0:-1, 0:-1] = adj
        return new_adj

    @staticmethod
    def add_node_attr(attr, new_row):
        return torch.cat([attr, new_row.unsqueeze(0)], 0)

    def add_zeros_row_batch(self, h):
        return [torch.cat([h[i], torch.zeros(1, self.hidden_size).to(self.device)], 0) for i in range(len(h))]

    def add_zeros_hidden(self, h):
        return torch.cat([h, torch.zeros(1, self.hidden_size).to(self.device)], 0)

    def get_vertexes_state(self, g_in, v_idx, hidden_single):
        hidden_v = []
        for i, g in enumerate(g_in):
            hv = torch.zeros(self.hidden_size).to(self.device) if v_idx[i] >= g[0].shape[0] \
                else hidden_single[i][v_idx[i], :]
            hidden_v.append(hv)
        hidden_v = torch.cat(hidden_v, 0).reshape(-1, self.hidden_size)
        return hidden_v

    @staticmethod
    def get_node_ordering(adj, last_idx):
        last = -1 if last_idx == -1 else last_idx + 1
        if adj[:last_idx, :last_idx].shape[0] > 0:
            gr = nx.from_numpy_matrix(adj[:last, :last].cpu().numpy(), create_using=nx.DiGraph)
            node_order = list(nx.dfs_preorder_nodes(gr, source=0))
            return node_order

    def decode(self, z, y=None, stochastic=True):
        if self.conditional:
            assert (y is not None)
            z = torch.cat([z, y], dim=1)
        hidden_zero = self.tanh(self.from_latent(z.float()).float())
        n_graphs = len(z)
        # first node generated is the starting node
        g_batch = [[torch.zeros(1, 1).float().to(self.device), torch.zeros(1, self.v_types).float().to(self.device)]
                   for _ in range(n_graphs)]
        for g in g_batch:
            g[1][0, 0] = 1
        hidden_single = [torch.zeros(1, self.hidden_size).to(self.device) for _ in range(n_graphs)]
        hidden_v, hidden_single = self.message_to(g_batch, 0, self.gru_dec, hidden_single=hidden_single,
                                                  hidden_agg=hidden_zero)
        completed = [False for _ in range(n_graphs)]
        complete_vertexes, no_pred_nodes = [[[] for _ in range(n_graphs)] for _ in range(2)]
        current_nodes_in_g = [1 for _ in range(n_graphs)]  # start vertex is already in all graphs
        # generate vertexes and sample their type
        v_idx = 1  # vertex I'm adding now
        while min(current_nodes_in_g) < self.max_n_vert - 1:
            if sum(completed) == n_graphs:  # generated all graphs
                break
            else:
                current_graph_state = self.get_graph_state(g_batch, hidden_single, intermediate=True,
                                                           decode=True)
                v_preds = [np.setdiff1d(np.arange(cur), np.array(com))
                           for cur, com in zip(current_nodes_in_g, complete_vertexes)]
                v_preds = list(map(lambda v: np.max(v) if len(v) > 0 else -1, v_preds))
                h_v_preds = self.get_vertexes_state(g_batch, v_preds, hidden_single)
                h_pred_type = torch.cat([current_graph_state, h_v_preds], 1)
                v_type_scores = self.vertex_type(h_pred_type)
                if stochastic:
                    v_type_probs = F.softmax(v_type_scores, 1).detach()
                    v_type = [torch.multinomial(v_type_probs[i], 1)+2 for i in range(n_graphs)]
                    # [np.random.choice(range(2, self.v_types), p=v_type_probs[i]) for i in range(n_graphs)]
                    assert (t != 0 for t in v_type)
                    assert (t != 1 for t in v_type)
                else:
                    v_type = torch.argmax(v_type_scores, 1) + 2
                    v_type = v_type.flatten().tolist()
            new_attr_rows = [torch.zeros(self.v_types).float().to(self.device) for _ in range(n_graphs)]
            for i, g in enumerate(g_batch):
                if not completed[i]:
                    if current_nodes_in_g[i] == len(complete_vertexes[i]) and len(complete_vertexes[i]) > 1:
                        # all vertexes are complete (i.e. a correct graph has been generated)
                        v_type[i] = self.end_idx
                    if current_nodes_in_g[i] == self.max_n_vert - 2:  # last node must be end type
                        # graph have maximum number of vertices
                        v_type[i] = self.end_idx
                    new_attr_rows[i][v_type[i]] = 1
                    g[1] = self.add_node_attr(g[1], new_attr_rows[i])  # type already set
                    g[0] = self.add_node_adj(g[0])
                    g[0][v_preds[i], v_idx] = 1 if v_preds[i] != -1 else 0
                    current_nodes_in_g[i] += 1  # a node can have a single predecessor
                    if v_type[i] >= len(self.type_arity) - 1:
                        complete_vertexes[i].append(v_idx)
                        complete_vertexes[i] = list(set(complete_vertexes[i]))  # remove duplicates
                    # update list of completed vertexes if necessary
                    type_v_p = torch.nonzero(g[1][v_preds[i], :]).item()
                    type_v_pred = len(self.type_arity) - 1 if type_v_p >= len(self.type_arity) - 1 \
                        else type_v_p
                    n_edge_v_pred = len(torch.nonzero(g[0][v_preds[i], :]).flatten())
                    if n_edge_v_pred == self.type_arity[type_v_pred]:
                        complete_vertexes[i].append(v_preds[i])
                        complete_vertexes[i] = list(set(complete_vertexes[i]))
                    if v_type[i] == self.end_idx:  # connect all the leaves to the end vertex
                        leaf_row_idx = torch.where(~torch.any(g[1][:, :5], dim=1))[0].type(torch.LongTensor)
                        for leaf in leaf_row_idx:
                            g[0][leaf, -1] = 1 if leaf not in no_pred_nodes[i] else 0  # connect leaves to end type
                        completed[i] = True
                if current_nodes_in_g[i] > self.max_n_vert - 2:
                    assert (completed[i] is True)
                    continue
            hidden_single = self.add_zeros_row_batch(hidden_single)  # add zero hidden state for new node
            # this allows to consider the type of the vertex in its hidden state
            hidden_v, hidden_single = self.message_to(g_batch, v_idx, self.gru_dec, hidden_single=hidden_single)
            v_idx += 1
        for hs in hidden_single:
            del hs  # to save GPU memory
        return g_batch

    def loss(self, mu, sigma, g_true, beta=0.001):  # 0.005
        # teacher forcing
        z = self.reparameterize(mu.float(), sigma.float())
        n_graphs = len(z)
        if self.conditional:
            y = torch.cat([g[2].reshape(1, -1) for g in g_true], 0)
            z = torch.cat([z, y], dim=1)
        hidden_zero = self.tanh(self.from_latent(z.float()).float())
        # start reconstructing
        g_batch = [[torch.zeros(1, 1).float().to(self.device), torch.zeros(1, self.v_types).float().to(self.device)]
                   for _ in range(n_graphs)]
        completed = [False for _ in range(n_graphs)]
        for g in g_batch:
            g[1][0, 0] = 1  # start node type
        hidden_single = [torch.zeros(1, self.hidden_size).to(self.device) for _ in range(n_graphs)]
        hidden_v, hidden_single = self.message_to(g_batch, 0, self.gru_dec, hidden_single=hidden_single,
                                                  hidden_agg=hidden_zero)
        ll = 0  # log-likelihood
        n_vert = [g[0].shape[0] for g in g_true]
        complete_vertexes, current_nodes_in_g = [[[] for _ in range(n_graphs)], [1 for _ in range(n_graphs)]]
        for v_true in range(1, max(n_vert)):
            if sum(completed) == n_graphs:
                break
            # compute likelihood of adding true type of nodes
            true_types = [torch.nonzero(g[1][v_true, :]).flatten() if v_true < n_vert[i] else self.start_idx
                          for i, g in enumerate(g_true)]
            hidden_g = self.get_graph_state(g_batch, hidden_single, intermediate=True, decode=True)
            v_pred = [np.setdiff1d(np.arange(cur), np.array(com))
                      for cur, com in zip(current_nodes_in_g, complete_vertexes)]
            v_pred = list(map(lambda v: np.max(v) if len(v) > 0 else -1, v_pred))
            h_v_pred = self.get_vertexes_state(g_batch, v_pred, hidden_single)
            h_pred_type = torch.cat([hidden_g, h_v_pred], 1)
            types_score = self.vertex_type(h_pred_type)
            new_attr_rows = [torch.zeros(self.v_types).float().to(self.device) for _ in range(n_graphs)]
            vll = self.logsoftmax(types_score)
            for i, g in enumerate(g_batch):
                if completed[i]:
                    continue
                if true_types[i] != self.end_idx and true_types[i] != self.start_idx:
                    # log-likelihood of adding true node
                    nll_loss = self.negLogLikelihood(vll[i].unsqueeze(0), torch.Tensor(
                        [true_types[i] - 2]).type(torch.LongTensor).to(self.device))
                    ll = ll + nll_loss  # update log-likelihood
                    current_nodes_in_g[i] += 1  # a node can have a single predecessor
                    if true_types[i] >= len(self.type_arity) - 1:
                        complete_vertexes[i].append(v_true)
                        complete_vertexes[i] = list(set(complete_vertexes[i]))  # remove duplicates
                    # update list of completed vertexes if necessary
                    type_v_p = torch.nonzero(g[1][v_pred[i], :]).item()
                    type_v_pred = len(self.type_arity) - 1 if type_v_p >= len(self.type_arity) - 1 \
                        else type_v_p
                    n_edge_v_pred = len(torch.nonzero(g[0][v_pred[i], :]).flatten())
                    if n_edge_v_pred == self.type_arity[type_v_pred]:
                        complete_vertexes[i].append(v_pred[i])
                        complete_vertexes[i] = list(set(complete_vertexes[i]))
                if true_types[i] != self.start_idx:
                    new_attr_rows[i][true_types[i]] = 1
                    g[1] = self.add_node_attr(g[1], new_attr_rows[i])
                    g[0] = self.add_node_adj(g[0])
                    g[0][v_pred[i], v_true] = 1 if v_pred[i] != -1 else 0
                    hidden_single[i] = self.add_zeros_hidden(hidden_single[i])
                if true_types[i] == self.end_idx:
                    leaf_row_idx = torch.where(~torch.any(g[1][:, :5], dim=1))[0].type(torch.LongTensor)
                    g[0][leaf_row_idx, -1] = 1
                    current_nodes_in_g[i] += 1
                    completed[i] = True
                    complete_vertexes[i].append(v_true)
                    complete_vertexes[i] = list(set(complete_vertexes[i]))  # remove duplicates
                if current_nodes_in_g[i] == len(complete_vertexes[i]) and len(complete_vertexes[i]) > 1:
                    assert (true_types[i] in [0, 1, len(self.type_arity) - 1])
                    assert (current_nodes_in_g[i] == n_vert[i])
                    completed[i] = True
            # here we already know the type and connectivity (i.e. all about the newly inserted vertex)
            hidden_v, hidden_single = self.message_to(g_batch, v_true, self.gru_dec, hidden_single=hidden_single)
        # kl-divergence
        kld = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        for hs in hidden_single:
            del hs  # to save GPU memory
        return ll + beta * kld, ll, kld

    def encode_decode(self, g_in):
        mu, sigma = self.encode(g_in)
        z = self.reparameterize(mu, sigma)
        return self.decode(z)

    def forward(self, g_in):
        mu, sigma = self.encode(g_in)
        loss, _, _ = self.loss(mu, sigma, g_in)
        return loss

    def generate_sample(self, n):
        sample = torch.randn(n, self.latent_size).to(self.device)
        g_sampled = self.decode(sample)
        return g_sampled


class ConvLogicVAE(LogicVAE):
    # implements convolutional versions of LogicVAE encoder
    def __init__(self, v_types, layers=3, heads=None, arity=None, max_v=None, start_idx=0, end_idx=1, h_dim=501,
                 z_dim=56, bidirectional=False, gat=True, conditional=False, semantic_length=0, device=None):
        super(ConvLogicVAE, self).__init__(v_types, arity=arity, max_v=max_v, start_idx=start_idx, end_idx=end_idx,
                                           h_dim=h_dim, z_dim=z_dim, bidirectional=bidirectional, gat=True,
                                           conditional=conditional, semantic_length=semantic_length, device=device)
        self.n_layers = layers
        self.attentional = gat
        if self.attentional:
            heads = [3] * (layers - 1) + [4] if heads is None else heads
            assert (len(heads) == layers)
            self.gat = GAT(layers, heads, [v_types] + [h_dim]*layers, self.device).gat_layers
        else:
            self.conv = nn.ModuleList()
            self.conv.append(nn.Sequential(nn.Linear(v_types, h_dim), nn.ReLU(), ))
            for layer in range(1, layers):
                self.conv.append(nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), ))

    def get_vertex_feature(self, g, v_idx, hidden_single_forward_g=None, layer=0):
        if layer == 0:
            feat = g[1][v_idx, :].to(self.device)
            if feat.shape[0] < self.v_types:
                feat = torch.cat([feat] + [torch.zeros(self.v_types - feat.shape[0]).to(self.device)])
        else:
            assert (hidden_single_forward_g is not None)
            feat = hidden_single_forward_g[v_idx, :]
        return feat.unsqueeze(0).to(self.device)

    def get_graph_state(self, g_in, hidden_single_forward, decode=False, start=0, offset=0, intermediate=False,
                        last_nodes=None):
        hidden_graph = []
        max_n_nodes = max([g[0].shape[0] for g in g_in])
        for i, g in enumerate(g_in):
            n_nodes_g = g[0].shape[0]
            if len(hidden_single_forward[i].shape) > 2:
                hidden_single_forward[i] = hidden_single_forward[i].squeeze(0)
            hidden_g = torch.cat([hidden_single_forward[i][v_idx, :].unsqueeze(0)
                                  for v_idx in range(start, n_nodes_g - offset)]).unsqueeze(0)
            if n_nodes_g < max_n_nodes:
                hidden_g = torch.cat([hidden_g, torch.zeros(1, max_n_nodes - n_nodes_g, hidden_g.shape[2]).to(
                    self.device)], 1)  # [1, max_n_nodes, hidden_size]
            hidden_graph.append(hidden_g)
        # use as graph state the sum of node states
        hidden_graph = torch.cat(hidden_graph, 0).sum(1).to(self.device)  # [n_batch, hidden_size]
        if self.conditional and not intermediate:
            semantic_vects = [g[2].reshape(1, -1) for g in g_in]
            semantic_vects = torch.cat(semantic_vects, 0)
            hidden_graph = torch.cat([hidden_graph, semantic_vects], 1)
        return hidden_graph

    @staticmethod
    def get_vertex_degree(adj, v_idx):
        v_out_degree = torch.sum(adj[v_idx, :])
        v_in_degree = 1 if v_idx != 0 else 0
        v_degree = v_out_degree + v_in_degree
        return [v_out_degree, v_in_degree, v_degree]

    def _conv_propagate_to(self, g_in, v_idx, hidden_single_forward, layer=0):
        # send messages to v_idx and update its hidden state
        n_vert = [g[0].shape[0] for g in g_in]
        graphs_idx = [i for i, _ in enumerate(g_in) if n_vert[i] > v_idx]
        graphs = [g for i, g in enumerate(g_in) if n_vert[i] > v_idx]
        if len(graphs) == 0:
            return None, hidden_single_forward
        v_info = [ConvLogicVAE.get_vertex_degree(g[0], v_idx) for g in graphs]
        v_predecessors = [torch.nonzero(g[0][:, v_idx]).flatten() for g in graphs]
        v_children = [torch.nonzero(g[0][v_idx, :]).flatten() for g in graphs]
        v_neigh = [torch.cat([v_pred, v_child]) for v_pred, v_child in zip(v_predecessors, v_children)]
        neigh_info = [[ConvLogicVAE.get_vertex_degree(g[0], n) for n in v_neigh[i]] for i, g in enumerate(graphs)]
        pred_info = [[ConvLogicVAE.get_vertex_degree(g[0], p) for p in v_predecessors[i]] for i, g in enumerate(graphs)]
        if self.bidirectional:  # accept messages also from children
            if self.attentional:
                h_neigh = [torch.cat([self.get_vertex_feature(g, v_idx, hidden_single_forward[graphs_idx[i]], layer)] +
                                     [self.get_vertex_feature(g, n_idx, hidden_single_forward[graphs_idx[i]], layer)
                                      for n_idx in v_neigh[i]], dim=0) for i, g in enumerate(graphs)]
            else:
                h_neigh = [torch.cat([self.get_vertex_feature(g, v_idx, hidden_single_forward[graphs_idx[i]],
                                                              layer) / (v_info[i][2] + 1)] +
                                     [self.get_vertex_feature(g, n_idx, hidden_single_forward[graphs_idx[i]],
                                                              layer) / math.sqrt((n_inf[2] + 1) * v_info[i][2] + 1)
                                      for n_idx, n_inf in zip(v_neigh[i], neigh_info[i])], 0)
                           for i, g in enumerate(graphs)]
        else:  # accept messages only from predecessors
            if self.attentional:
                h_neigh = [torch.cat([self.get_vertex_feature(g, v_idx, hidden_single_forward[graphs_idx[i]], layer)] +
                                     [self.get_vertex_feature(g, p_idx, hidden_single_forward[graphs_idx[i]], layer)
                                      for p_idx in v_predecessors[i]], dim=0) for i, g in enumerate(graphs)]
            else:
                h_neigh = [torch.cat([self.get_vertex_feature(g, v_idx, hidden_single_forward[graphs_idx[i]],
                                                              layer) / (v_info[i][1] + 1)] +
                                     [self.get_vertex_feature(g, p_idx, hidden_single_forward[graphs_idx[i]],
                                                              layer) / math.sqrt((p_inf[0] + 1) * v_info[i][1] + 1)
                                      for p_idx, p_inf in zip(v_predecessors[i], pred_info[i])], 0)
                           for i, g in enumerate(graphs)]
        if self.attentional:
            h_self = [torch.cat([self.get_vertex_feature(g, v_idx, hidden_single_forward[graphs_idx[i]], layer)] *
                                h_neigh[i].shape[0], 0) for i, g in enumerate(graphs)]
            max_n_neigh = max([n.shape[0] for n in h_neigh])
            h_self = [torch.cat([h_s.to(self.device)] + [torch.zeros(max_n_neigh - len(h_s), h_s.shape[1]).to(
                self.device)], 0).unsqueeze(0) for h_s in h_self]
            h_self = torch.cat(h_self, 0).to(self.device)  # [batch, max_n_neigh, n_types]
            h_neigh = [torch.cat([h_n.to(self.device)] + [torch.zeros(max_n_neigh - len(h_n), h_n.shape[1]).to(
                self.device)], 0).unsqueeze(0) for h_n in h_neigh]
            h_neigh = torch.cat(h_neigh, 0).to(self.device)  # [batch, max_n_neigh, n_types]
            h_v = self.gat[layer](h_self, h_neigh)

        else:
            max_n_neigh = max([n.shape[0] for n in h_neigh])
            h_neigh = [torch.cat([h_n.to(self.device)] + [torch.zeros(max_n_neigh - len(h_n), h_n.shape[1]).to(
                self.device)], 0).unsqueeze(0) for h_n in h_neigh]
            h_neigh = torch.cat(h_neigh, 0).to(self.device)  # [batch, max_n_neigh, n_types]
            h_v = self.conv[layer](h_neigh.sum(1).to(self.device))  # [n_batch, hidden_size]
        for i, g in enumerate(graphs):
            hv = h_v[i:i+1].clone()
            if len(hv.shape) > 2:
                if h_v[i:i+1].shape[2] > hidden_single_forward[graphs_idx[i]].shape[1]:
                    pad = nn.ConstantPad2d((0,
                                            h_v[i:i+1].shape[2] - hidden_single_forward[graphs_idx[i]].shape[1], 0, 0),
                                           0)
                    hidden_single_forward[graphs_idx[i]] = pad(hidden_single_forward[graphs_idx[i]].clone())
                elif h_v[i:i+1].shape[2] < hidden_single_forward[graphs_idx[i]].shape[1]:
                    pad = nn.ConstantPad2d((0,
                                            hidden_single_forward[graphs_idx[i]].shape[1] - h_v[i:i+1].shape[2], 0, 0),
                                           0)
                    hv = pad(h_v[i:i+1].clone())
            hidden_single_forward[graphs_idx[i]][v_idx, :] = hv
        return h_v, hidden_single_forward

    def encode(self, g_in):
        if type(g_in) is not list:
            g_in = [g_in]
        prop_order = range(self.max_n_vert)
        n_vars = [g[0].shape[0] for g in g_in]
        hidden_single_forward = [torch.zeros(n, self.hidden_size).to(self.device) for i, n in enumerate(n_vars)]
        for layer in range(self.n_layers):
            for v in prop_order:
                h_v, hidden_single_forward = self._conv_propagate_to(g_in, v, hidden_single_forward, layer)
        for i, n in enumerate(n_vars):
            hidden_single_forward[i] = hidden_single_forward[i][:n, :self.hidden_size]
        # exclude start and end node
        hidden_g = self.get_graph_state(g_in, hidden_single_forward, start=1, offset=1, intermediate=False)
        mu, sigma = self.mlp_mean(hidden_g.float()), self.mlp_std(hidden_g.float())
        for hs in hidden_single_forward:
            del hs  # to save GPU memory
        return mu, sigma
