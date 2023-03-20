#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def general_settings(parser):
    parser.add_argument('--data', default='data/data-dvae-5-var-big/', type=str,
                        help='dataset name')  # exclude .pickle suffix and folder prefix
    parser.add_argument('--data_folder', type=str, default='data', help='directory where data are stored')
    parser.add_argument('--p_leaf', type=float, default=0.5, help='leaf probability in formula generation')
    parser.add_argument('--max_depth', type=int, default=100, help='max formula depth in formula generation')
    parser.add_argument('--max_nodes', type=int, default=25, help='max number of nodes in the formula parse tree')
    parser.add_argument('--n_vars', default=[5], help='number of variables in formula generation')
    parser.add_argument('--n_graphs', type=int, default=5000,  # 5000
                        help='number of graphs (for each number of variable) to generate')
    parser.add_argument('--max_n_vars', default=5, type=int, help='max number of variables supported by the model')
    parser.add_argument('--run_folder', type=str, default='run', help='directory to save checkpoint')
    parser.add_argument('--result_folder', type=str, default='results', help='directory to store results')
    parser.add_argument('--model_name', type=str, default='dvae-cnv-4-var-conditional',
                        help='name of the attempted model')


def model_settings(parser):
    parser.add_argument('--load', type=str,
                        default=None,
                        help='checkpoint file to load')  # should eventually include run/ sub-folder
    parser.add_argument('--train', default=True, type=bool, help='whether to train the model')
    parser.add_argument('--restore_train', default=False, type=bool,
                        help='whether to continue training a pre-trained model')
    parser.add_argument('--test', default=False, type=bool, help='whether to test the model')
    parser.add_argument('--encode_times', type=int, default=10, help='number of encoding times while testing')
    parser.add_argument('--decode_times', type=int, default=10, help='number of decoding times while testing')
    parser.add_argument('--semantic_length', default=100, type=int, help='length of the semantic context vector')


def recover_settings(parser):
    parser.add_argument('--hidden_size', type=int, default=200, help='dimension of hidden node states')
    parser.add_argument('--latent_size', type=int, default=56, help='dimension of latent vectors')
    parser.add_argument('--var_indexes', default=True, type=bool,
                        help='whether to consider variable indexes in the dataset')
    parser.add_argument('--beta', default=0.001, help='weight of KL divergence in loss computation')
    parser.add_argument('--bidirectional', default=True, type=bool, help='whether to use bidirectional GRU')
    parser.add_argument('--v_id', default=True, type=bool,
                        help='whether to give unique identifier to each vertex in the graph')
    parser.add_argument('--encode_gcn', default=False, type=bool,
                        help='whether to use graph convolutions to encode input graphs')
    parser.add_argument('--gat', default=False, type=bool, help='whether to use attention in GCN')
    parser.add_argument('--gcn_layers', default=2, type=int, help='number of GCN layers in LogicVAE encoding')
    parser.add_argument('--conditional', default=False, type=bool, help='whether to condition on semantic vector the AE')


def optimization_settings(parser):
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  # 1e-3
    parser.add_argument('--n_epochs', type=float, default=501, help='number of epochs of training')  # 301
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size of batches during training (1 is online training)')
