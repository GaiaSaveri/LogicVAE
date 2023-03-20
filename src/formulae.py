#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy.random as rnd

import src.proplogic


class PropFormula:
    def __init__(self, leaf_prob, inner_node_prob, max_depth):
        if inner_node_prob is None:
            inner_node_prob = [0.35, 0.35, 0.30]
        self.leaf_prob = leaf_prob
        self.inner_node_prob = inner_node_prob
        self.node_types = ["and", "or", "not"]
        self.max_depth = max_depth
        self.n_vars = None

    def sample(self, n_vars):
        self.n_vars = n_vars
        return self._sample_internal_node(n_vars)

    def bag_sample(self, n_formulae, n_vars):
        formulae = []
        for _ in range(n_formulae):
            phi = self.sample(n_vars)
            formulae.append(phi)
        return formulae

    def _sample_internal_node(self, n_vars):
        node = None
        node_type = rnd.choice(self.node_types, p=self.inner_node_prob)
        while True:
            if node_type == "not":
                child = self._sample_node(n_vars)
                node = src.proplogic.Negation(child)
            else:
                left_child = self._sample_node(n_vars)
                right_child = self._sample_node(n_vars)
                if node_type == "and":
                    node = src.proplogic.Conjunction(left_child, right_child)
                elif node_type == "or":
                    node = src.proplogic.Disjunction(left_child, right_child)
            if (node is not None) and (node.depth() < self.max_depth):
                return node

    def _sample_node(self, n_vars):
        if rnd.rand() < self.leaf_prob:
            var_idx = rnd.randint(n_vars)
            return src.proplogic.LogicVar(var_idx)
        else:
            return self._sample_internal_node(n_vars)
