#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pickle
import os
import collections
import random
import math

from src.tests import check_syntactic_validity
import src.data_utils


def decode_most_common(model, z_list, decode_times=20):
    decoded = []
    for z in z_list:
        dec_z = []
        dic_z = dict()
        for _ in range(decode_times):
            g_rec = model.decode(z)[0]
            g_checks = [g_rec[0][1:-1, 1:-1], g_rec[1][1:-1, 2:]]
            val = check_syntactic_validity(g_checks)
            if val:
                dec_phi = src.data_utils.from_output_to_formula(g_checks[0], g_checks[1])
                dec_z.append(str(dec_phi))
                dic_z[str(dec_phi)] = dec_phi
        if len(dec_z) > 0:  # at least a valid decoding
            cnt = collections.Counter(dec_z)
            decoded.append(dic_z[cnt.most_common(1)[0][0]])
    return decoded


def linear_interpolation(model, model_name, data_folder, n_points=100, simplified=True):
    model.eval()
    folder_files = os.listdir(data_folder)
    test_files = [i for i in folder_files if i.startswith('validation')]
    with open(data_folder + os.path.sep + test_files[0], 'rb') as f:
        test_data = pickle.load(f)
    attr_idx = 2 if (simplified and len(test_data[0]) > 2) else 1
    test_data = [[d[0].to(model.device), d[attr_idx].to(model.device)] for d in test_data][20:]
    data = src.data_utils.PropFormulaeLoader.get_dvae_input(test_data)
    all_interp, all_dist = [[], []]
    for i in range(0, len(data)-1, 2):
        z_1, _ = model.encode([data[i]])
        z_2, _ = model.encode([data[i+1]])
        z_j, interp_j = [[], []]
        for j in range(0, n_points + 1):
            zj = z_1 + (z_2 - z_1) / n_points * j
            z_j.append(zj)
        all_interp.append(decode_most_common(model, z_j))
        all_dist.append(torch.norm(z_1 - z_2))
    with open('results' + os.path.sep + model_name + os.path.sep + "linear_latent_" + model_name + ".pickle", "wb") \
            as f:
        pickle.dump(all_interp, f)
    with open('results' + os.path.sep + model_name + os.path.sep + "linear_latent_dist_" + model_name + ".pickle",
              "wb") as f:
        pickle.dump(all_dist, f)


def spherical_interpolation(model, model_name, data_folder, n_points=36, simplified=True):
    model.eval()
    folder_files = os.listdir(data_folder)
    test_files = [i for i in folder_files if i.startswith('validation')]
    with open(data_folder + os.path.sep + test_files[0], 'rb') as f:
        test_data = pickle.load(f)
    attr_idx = 2 if (simplified and len(test_data[0]) > 2) else 1
    test_data = [[d[0].to(model.device), d[attr_idx].to(model.device)] for d in test_data][:20]
    data = src.data_utils.PropFormulaeLoader.get_dvae_input(test_data)
    all_interp, all_omega = [[], []]
    for i, g in enumerate(data):
        z_i, _ = model.encode([g])
        norm_i = torch.norm(z_i)
        z_j = torch.ones_like(z_i)
        dim_to_change = random.randint(0, z_i.shape[1] - 1)
        z_j[0, dim_to_change] = -(z_i[0, :].sum() - z_i[0, dim_to_change]) / z_i[0, dim_to_change]
        z_j = z_j / torch.norm(z_j) * norm_i
        omega = torch.acos(torch.dot(z_i.flatten(), z_j.flatten()))
        z_x = []
        for x in range(0, n_points + 1):
            theta = 2 * math.pi / n_points * x
            zx = z_i * torch.cos(torch.Tensor([theta])) + z_j * torch.sin(torch.Tensor([theta]))
            z_x.append(zx)
        all_interp.append(decode_most_common(model, z_x))
        all_omega.append(omega)
    with open('results' + os.path.sep + model_name + os.path.sep + "spherical_latent_" + model_name + ".pickle", "wb") \
            as f:
        pickle.dump(all_interp, f)
    with open('results' + os.path.sep + model_name + os.path.sep + "spherical_latent_omega_" + model_name + ".pickle",
              "wb") as f:
        pickle.dump(all_omega, f)
