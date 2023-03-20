#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
from sklearn.decomposition import PCA
import collections
from scipy.spatial.distance import cdist, euclidean

from src.tests import check_syntactic_validity
import src.data_utils
from src.propkernel import *
from src.proplogic import *
from src.formulae import *


def get_latent_dist(model, train_data, batch_size=32):
    model.eval()
    z = []
    data = src.data_utils.PropFormulaeLoader.get_dvae_input(train_data)
    train_batches = src.data_utils.PropFormulaeLoader.divide_batches(data, batch_size, len(data))
    for i, g in enumerate(train_batches):
        mu, _ = model.encode(g)
        mu = mu.cpu().detach().numpy()
        z.append(mu)
    return np.concatenate(z, 0)


def validity_uniqueness_novelty(model, model_name, n_samples=1000, decode_times=10, data_folder=None, train_range=True,
                                simplified=False):
    find_novelty = False
    phi_train, z_mean, z_std = [None for _ in range(3)]
    if data_folder is not None:
        find_novelty = True
        folder_files = os.listdir(data_folder)
        train_files = [i for i in folder_files if i.startswith('training')]
        with open(data_folder + os.path.sep + train_files[0], 'rb') as f:
            train_data = pickle.load(f)
        attr_idx = 2 if (simplified and len(train_data[0]) > 2) else 1
        phi_train = set([str(src.data_utils.from_output_to_formula(d[0], d[attr_idx])) for d in train_data])
        if train_range:
            z_train = get_latent_dist(model, train_data)
            z_mean, z_std = [torch.FloatTensor(z_train.mean(0)).to(model.device),
                             torch.FloatTensor(z_train.std(0)).to(model.device)]
    # prior validity
    sample = torch.randn(n_samples, model.latent_size).to(model.device)
    valid = 0
    single_valid, single_unique, single_novel = [[] for _ in range(3)]
    phi_gen = set()
    for i in range(n_samples):
        if train_range:
            sample = sample * z_std + z_mean  # move to train latent range
        valid_i, novel_i = [0 for _ in range(2)]
        phi_gen_i = set()
        for _ in range(decode_times):
            g_rec = model.decode(sample[i].unsqueeze(0))[0]
            g_checks = [g_rec[0][1:-1, 1:-1], g_rec[1][1:-1, 2:]]
            val = check_syntactic_validity(g_checks)
            valid, valid_i = [x + int(val) for x in [valid, valid_i]]
            if val:
                # uniqueness
                phi_dec_i = src.data_utils.from_output_to_formula(g_checks[0], g_checks[1])
                phi_gen_i.add(str(phi_dec_i))
                phi_gen.add(str(phi_dec_i))
            if find_novelty:
                # novelty
                phi_unique_i = phi_train.union(phi_gen_i)
                novel_i += len(phi_unique_i) - len(phi_train)
        single_valid.append(valid_i / decode_times)
        s_unique = len(phi_gen_i) / valid_i if valid_i != 0 else 0
        single_unique.append(s_unique)
        single_novel.append(novel_i / decode_times)
        file = open('results' + os.path.sep + model_name + os.path.sep + "vae_ability_single_" + model_name + ".txt",
                    "a")
        file.write("Validity / Uniqueness / Novelty: %f %f %f\n" % (single_valid[i], single_unique[i], single_novel[i]))
        file.close()
    valid = valid / (n_samples * decode_times)
    unique = len(phi_gen) / (valid * n_samples * decode_times) if valid > 0 else 0
    novelty = (len(phi_train.union(phi_gen)) - len(phi_train)) / (valid * n_samples * decode_times) if valid > 0 else 0
    return valid, unique, novelty


def cvae_abilities(model, n_vars, semantic_length=100, n_samples=300, n_z=100, decode_times=10):
    phi_gen = PropFormula(0.5, None, 100)
    n_components = semantic_length
    enc_phis = phi_gen.bag_sample(5000, n_vars)
    test_phis = phi_gen.bag_sample(n_samples, n_vars)
    # sample signals
    meas = PropUniformMeasure(n_vars, 'cpu')
    # instantiate kernel
    kernel = PropKernel(meas, samples=None, varn=n_vars, fuzzy=False)
    gram_train = kernel.compute_bag_bag(enc_phis, enc_phis)
    gram_test = kernel.compute_bag_bag(test_phis, enc_phis).cpu().numpy()
    print('Ground Truth Mean Kernel: ', np.mean(gram_test), np.std(gram_test))

    kpca = PCA(n_components=n_components)
    kpca.fit(gram_train)
    reduced_test = kpca.transform(gram_test)
    reduced_test = torch.from_numpy(reduced_test)
    print('Ground Truth Mean Distance: ', np.mean(cdist(reduced_test, reduced_test)),
          np.std(cdist(reduced_test, reduced_test)))
    valid = 0
    single_valid, single_unique = [[] for _ in range(2)]
    mean_kernels, mean_distances = [[], []]
    for i in range(n_samples):  # SEMANTIC INFO
        sample = torch.randn(n_z, model.latent_size).to(model.device)  # z for semantic vector i
        valid_i = 0
        phi_i = []
        phi_gen_i = set()
        list_for_kernel = []
        for j in range(len(sample)):
            valid_ij = 0
            phi_gen_ij = set()
            phi_ij = []
            phi_dict = dict()
            for _ in range(decode_times):
                g_rec = model.decode(sample[j].unsqueeze(0), y=reduced_test[i])[0]
                g_checks = [g_rec[0][1:-1, 1:-1], g_rec[1][1:-1, 2:]]
                val = check_syntactic_validity(g_checks)
                valid, valid_i, valid_ij = [x + int(val) for x in [valid, valid_i, valid_ij]]
                if val:
                    # uniqueness
                    phi_dec_ij = src.data_utils.from_output_to_formula(g_checks[0], g_checks[1])
                    phi_gen_ij.add(str(phi_dec_ij))
                    phi_gen_i.add(str(phi_dec_ij))
                    phi_i.append(phi_dec_ij)
                    phi_ij.append(phi_dec_ij)
                    phi_dict[str(phi_dec_ij)] = phi_dec_ij
            if len(phi_ij) > 0:
                cnt = collections.Counter(phi_dict.keys())
                list_for_kernel.append(phi_dict[cnt.most_common(1)[0][0]])
            single_valid.append(valid_i / decode_times)
            s_unique = len(phi_gen_i) / valid_i if valid_i != 0 else 0
            single_unique.append(s_unique)
        if len(list_for_kernel[i]) > 0:
            # phi_i must be used to compute kernel (TAKE MOST COMMON) and distance
            # distance among most decoded formulae for a single semantic vector and different syntactic z
            kernel_i = kernel.compute_bag_bag(list_for_kernel, list_for_kernel).cpu().numpy()
            kernel_i_row_mean = np.mean(kernel_i, axis=1)
            mean_kernels.append(np.mean(kernel_i_row_mean))
            # distance between most decoded decoded formulae and semantic embedding
            gram_most_decoded = kernel.compute_bag_bag(list_for_kernel, enc_phis)
            reduced_decode = kpca.transform(gram_most_decoded)
            distance_i = np.mean([euclidean(reduced_test[i], r) for r in reduced_decode])
            mean_distances.append(distance_i)
    print('Mean Kernel: ', np.mean(mean_kernels), np.std(mean_kernels))
    print('Mean Distance: ', np.mean(mean_distances), np.std(mean_distances))
    valid = valid / (n_samples * decode_times * n_z)
    print('Validity: ', valid)
