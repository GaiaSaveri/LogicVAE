#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import itertools

import src.proplogic


class Measure:
    def sample(self, samples=100000, varn=2, points=100):
        # Must be overridden
        pass


class PropUniformMeasure(Measure):
    def __init__(self, varn, device, max_samples=100000):
        self.varn = varn
        self.device = device
        self.max_samples = max_samples
        self.samples = None

    def sample(self, samples=None, varn=None, points=None):
        if 2**self.varn > self.max_samples or samples:
            n_samples = self.max_samples if samples is None else samples
            x = torch.cat([torch.bernoulli(torch.Tensor([0.5 for _ in range(self.varn)]))
                           for _ in range(n_samples)]).reshape(-1, self.varn)
        else:
            x = torch.Tensor(list(itertools.product([0, 1], repeat=self.varn))).to(self.device)
        self.samples = x.shape[0]
        return x


class PropKernel:
    def __init__(
        self,
        measure,
        normalize=True,
        exp_kernel=True,
        sigma2=0.2,
        samples=None,
        varn=2,
        signals=None,
        fuzzy=False,
    ):
        self.traj_measure = measure
        self.exp_kernel = exp_kernel
        self.normalize = normalize
        self.sigma2 = sigma2
        self.varn = varn
        self.fuzzy = fuzzy
        if signals is not None:
            self.signals = signals
            self.samples = signals.shape[0]
        else:
            self.signals = measure.sample(samples=samples, varn=varn)
            self.samples = samples if samples else measure.samples

    def compute(self, phi1, phi2):
        return self.compute_one_one(phi1, phi2)

    def compute_one_one(self, phi1, phi2):
        phis1: list = [phi1]
        phis2: list = [phi2]
        ker = self.compute_bag_bag(phis1, phis2)
        return ker[0, 0]

    def compute_bag(self, phis, return_sat=True):
        sats, selfk = self._evaluate_sat(phis)
        kernel_matrix = self._compute_kernel(sats, sats, selfk, selfk)
        len0 = None
        if return_sat:
            return kernel_matrix.cpu(), sats, selfk, len0
        else:
            return kernel_matrix.cpu()

    def compute_one_bag(self, phi1, phis2, return_sat=False):
        phis1 = [phi1]
        return self.compute_bag_bag(phis1, phis2, return_sat)

    def compute_bag_bag(self, phis1, phis2, return_sat=False):
        sats1, selfk1 = self._evaluate_sat(phis1)
        sats2, selfk2 = self._evaluate_sat(phis2)
        kernel_matrix = self._compute_kernel(sats1, sats2, selfk1, selfk2)
        if return_sat:
            return kernel_matrix.cpu(), sats1, sats2, selfk1, selfk2
        else:
            return kernel_matrix.cpu()

    def compute_one_from_sat(self, phi, sats, sat_self, return_sat=False):
        phis = [phi]
        return self.compute_bag_from_sat(phis, sats, sat_self, return_sat)

    def compute_bag_from_sat(self, phis, sats, sat_self, return_sat=False):
        sats1, selfk1 = self._evaluate_sat(phis)
        kernel_matrix = self._compute_kernel(sats1, sats, selfk1, sat_self)
        if return_sat:
            return kernel_matrix.cpu(), sats1, selfk1
        else:
            return kernel_matrix.cpu()

    def _evaluate_sat(self, phis):
        n = self.samples
        k = len(phis)
        sats = torch.zeros((k, n), device=self.traj_measure.device)
        self_kernels = torch.zeros((k, 1), device=self.traj_measure.device)
        for i, phi in enumerate(phis):
            if not self.fuzzy:
                sat = phi.boolean(self.signals).float().squeeze()
                sat[sat == 0.0] = -1.0
            else:
                sat = phi.fuzzy(self.signals).float().squeeze()
                sat = 2*sat - 1  # reparameterize to be in [-1, 1]
            if n == 1:
                sat = sat.reshape(1)
            self_kernels[i] = sat.dot(sat) / n
            sats[i, :] = sat
        return sats, self_kernels

    def evaluate_sat(self, phis):
        return self._evaluate_sat(phis)

    def _compute_kernel(self, sats1, sats2, selfk1, selfk2):
        kernel_matrix = torch.tensordot(sats1, sats2, [[1], [1]])
        kernel_matrix = kernel_matrix / self.samples
        if self.normalize:
            kernel_matrix = self._normalize(kernel_matrix, selfk1, selfk2)
        if self.exp_kernel:
            kernel_matrix = self._exponentiate(kernel_matrix, selfk1, selfk2)
        return kernel_matrix

    def compute_kernel(self, sats1, sats2, selfk1, selfk2):
        return self._compute_kernel(sats1, sats2, selfk1, selfk2)

    @staticmethod
    def _normalize(kernel_matrix, selfk1, selfk2):
        normalize = torch.sqrt(torch.matmul(selfk1, torch.transpose(selfk2, 0, 1)))
        kernel_matrix = kernel_matrix / normalize
        return kernel_matrix

    def _exponentiate(self, kernel_matrix, selfk1, selfk2, sigma2=None):
        if sigma2 is None:
            sigma2 = self.sigma2
        if self.normalize:
            # selfk is (1.0^2 + 1.0^2)
            selfk = 2.0
        else:
            k1 = selfk1.size()[0]
            k2 = selfk2.size()[0]
            selfk = (selfk1 * selfk1).repeat(1, k2) + torch.transpose(
                selfk2 * selfk2, 0, 1
            ).repeat(k1, 1)
        return torch.exp(-(selfk - 2 * kernel_matrix) / (2 * sigma2))


def variable_naming(phi, idx_list):
    if type(phi) is not src.proplogic.LogicVar:
        if type(phi) is src.proplogic.Negation:
            phi.child, idx_list = variable_naming(phi.child, idx_list)
        else:
            phi.left_child, idx_list = variable_naming(phi.left_child, idx_list)
            phi.right_child, idx_list = variable_naming(phi.right_child, idx_list)
    else:
        phi = src.proplogic.LogicVar(idx_list[0])
        idx_list = idx_list[1:]
    return phi, idx_list


def from_prob_to_fuzzy_ass(prob_list, traj):
    # assume prob_list is a tensor [# leaves, # vars]
    # traj is a tensor [# signals, # vars]
    prob_l = prob_list.transpose(0, 1)
    assert (prob_l.shape[0] == traj.shape[1])
    fuzzy_ass = traj.matmul(prob_l)
    # output shape should be [# signals, # leaves]
    assert (fuzzy_ass.shape[0] == traj.shape[0]) and (fuzzy_ass.shape[1] == prob_list.shape[0])
    return fuzzy_ass


def evaluate_sat_from_signals(sig_tensor, phi):
    # assume phi has ordered variable numbering
    # i.e. as many variables as the number of leaves in its syntactic tree
    n_signals = sig_tensor.shape[0]
    sats = 2 * phi.fuzzy(sig_tensor).squeeze() - 1
    self_k = sats.dot(sats) / n_signals
    return sats.view(1, n_signals), self_k
