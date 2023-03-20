#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import optim
import argparse
import time


from src.data_utils import PropFormulaeLoader, PropFormulaeIsoLoader
from src.LogicVAE import LogicVAE, ConvLogicVAE
from src.train_dvae import train
from src.test_dvae import test
from src.utils import load, execution_time, get_semantic_embedding, from_output_to_formula
from options import *
from src.formulae import *
from src.propkernel import *
from src.latent_visualization import *
from src.vae_abilities import validity_uniqueness_novelty, cvae_abilities


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on: ', device)
    # TODO: check path of these directories when refactoring
    if not os.path.exists(os.path.join(os.getcwd(), args.result_folder + os.path.sep + args.model_name)):
        os.makedirs(args.result_folder + os.path.sep + args.model_name)
    if not os.path.exists(os.path.join(os.getcwd(), args.run_folder + os.path.sep + args.model_name)):
        os.makedirs(args.run_folder + os.path.sep + args.model_name)
    with open(args.run_folder + os.path.sep + args.model_name + os.path.sep + 'parameters.pickle', 'wb') as out:
        pickle.dump(vars(args), out)
    # define model
    train_loader = PropFormulaeLoader(args.p_leaf, args.max_depth, args.n_vars, args.n_graphs, args.max_nodes,
                                      device, args.data)
    enc_phis, n_data, gram_train = [None for _ in range(3)]
    max_v = args.max_nodes + 2
    v_types = int(args.max_n_vars + 3 + 2) if args.var_indexes else 4 + 2
    if not args.encode_gcn:
        model = LogicVAE(v_types=v_types, max_v=max_v, h_dim=args.hidden_size, z_dim=args.latent_size,
                         bidirectional=args.bidirectional, v_id=args.v_id, conditional=args.conditional,
                         semantic_length=args.semantic_length, device=device)
    else:
        model = ConvLogicVAE(v_types=v_types, layers=args.gcn_layers, max_v=max_v, h_dim=args.hidden_size,
                             z_dim=args.latent_size, gat=args.gat,
                             bidirectional=args.bidirectional, conditional=args.conditional,
                             semantic_length=args.semantic_length, device=device)
    if args.conditional:
        # sample formulae to compute semantic encodings
        phi_gen = PropFormula(args.p_leaf, None, args.max_depth)
        n_components = args.semantic_length
        enc_phis = phi_gen.bag_sample(5000, args.n_vars[0])
        # sample signals
        meas = PropUniformMeasure(args.n_vars[0], device)
        # instantiate kernel
        kernel = PropKernel(meas, samples=None, varn=args.n_vars[0], fuzzy=False)
        gram_train = kernel.compute_bag_bag(enc_phis, enc_phis)

    model.to(device)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    if args.train:
        checkpoint = load(args, model, device, optimizer) if args.restore_train else None
        train(args, model, train_loader, optimizer, enc_phis=enc_phis, gram_train=gram_train,
              checkpoint=checkpoint)
    if args.test:
        data_loader = PropFormulaeLoader(args.p_leaf, args.max_depth, args.n_vars, args.n_graphs, args.max_nodes,
                                         device, args.data)
        _ = load(args, model, device, optimizer) if not args.train else None
        print("\n...Loading test data...")
        test_batches = data_loader.get_data(kind='test', simplified=(not args.var_indexes))
        if args.conditional:
            vector = args.conditional
            n_components = args.hidden_size if args.semantic_init else args.semantic_length
            test_phi_list = [from_output_to_formula(d[0][1:-1, 1:-1], d[1][1:-1, 2:]) for d in test_batches]
            test_phi_inits = get_semantic_embedding(args.n_vars[0], enc_phis, test_phi_list,
                                                    n_components, model.device, gram_train=gram_train)
            get_test_kernel = [d.append(test_phi_inits[d_idx, :]) for d_idx, d in enumerate(test_batches)]
        print("...loaded")
        test_start = time.time()
        test_loss, stats_perf = test(model, args, test_batches, data_loader=data_loader)
        test_end = time.time()
        test_h, test_m, test_s = execution_time(test_start, test_end)
        if args.conditional:
            cvae_abilities(model, args.n_vars[0])
        else:
            validity, uniqueness, novelty = validity_uniqueness_novelty(
                model, args.model_name, data_folder=args.data, simplified=(not args.var_indexes), train_range=False)
            spherical_interpolation(model, args.model_name, args.data)
            print("Validity/Uniqueness/Novelty: {:.4f} / {:.4f} / {:.4f}".format(validity, uniqueness, novelty))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    general_settings(parser)
    model_settings(parser)
    arg = parser.parse_args()
    if arg.load or arg.restore_train:
        with open(arg.run_folder + os.path.sep + arg.model_name + os.path.sep + 'parameters.pickle', 'rb') as f:
            ar = pickle.load(f)
        for k in ['hidden_size', 'latent_size', 'var_indexes', 'beta', 'lr', 'batch_size',
                  'bidirectional', 'v_id', 'constrained', 'batch_train', 'kl_annealing', 'encode_gcn',
                  'gcn_layers', 'conditional', 'gat']:
            parser.add_argument('--' + k, default=ar[k])
    else:
        recover_settings(parser)
        optimization_settings(parser)
    main(args=parser.parse_args())
