#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class LogicNode:
    def __init__(self):
        pass

    def __str__(self):
        pass

    def depth(self):
        pass

    def _boolean(self, x):
        pass

    def boolean(self, x):
        return self._boolean(x)

    def _fuzzy(self, p):
        pass

    def fuzzy(self, p):
        return self._fuzzy(p)


class LogicVar(LogicNode):
    def __init__(self, var_idx):
        super().__init__()
        self.var_index = var_idx
        self.name = "x_" + str(self.var_index)

    def __str__(self):
        return self.name

    def depth(self):
        return 0

    def _boolean(self, x):
        xj = x[:, self.var_index]
        xj = xj.view(xj.size()[0], 1)
        return xj.type(dtype=torch.int64)

    def _fuzzy(self, p):
        pj = p[:, self.var_index]
        pj = pj.view(pj.size()[0], 1)
        return pj.type(dtype=torch.float64)


class Negation(LogicNode):
    def __init__(self, child):
        super().__init__()
        self.child = child
        self.name = "not"

    def __str__(self):
        s = "not ( " + self.child.__str__() + " )"
        return s

    def depth(self):
        return 1 + self.child.depth()

    def _boolean(self, x):
        z = torch.abs(self.child._boolean(x).type(dtype=torch.int64) - 1)
        return z

    def _fuzzy(self, p):
        z = 1 - self.child._fuzzy(p).type(dtype=torch.float64)
        return z


class Conjunction(LogicNode):
    def __init__(self, left_child, right_child):
        super().__init__()
        self.left_child = left_child
        self.right_child = right_child
        self.name = "and"

    def __str__(self):
        s = "( " + self.left_child.__str__() + " and " + self.right_child.__str__() + " )"
        return s

    def depth(self):
        return 1 + max(self.left_child.depth(), self.right_child.depth())

    def _boolean(self, x):
        z1 = self.left_child._boolean(x)
        z2 = self.right_child._boolean(x)
        size = min(z1.size()[1], z2.size()[1])
        z1 = z1[:, :size]
        z2 = z2[:, :size]
        z = torch.logical_and(z1, z2)
        return z

    def _fuzzy(self, p):
        z1 = self.left_child._fuzzy(p)
        z2 = self.right_child._fuzzy(p)
        size = min(z1.size()[1], z2.size()[1])
        z1 = z1[:, :size]
        z2 = z2[:, :size]
        z = torch.minimum(z1, z2)
        return z


class Disjunction(LogicNode):
    def __init__(self, left_child, right_child):
        super().__init__()
        self.left_child = left_child
        self.right_child = right_child
        self.name = "or"

    def __str__(self):
        s = "( " + self.left_child.__str__() + " or " + self.right_child.__str__() + " )"
        return s

    def depth(self):
        return 1 + max(self.left_child.depth(), self.right_child.depth())

    def _boolean(self, x):
        z1 = self.left_child._boolean(x)
        z2 = self.right_child._boolean(x)
        size = min(z1.size()[1], z2.size()[1])
        z1 = z1[:, :size]
        z2 = z2[:, :size]
        z = torch.logical_or(z1, z2)
        return z

    def _fuzzy(self, p):
        z1 = self.left_child._fuzzy(p)
        z2 = self.right_child._fuzzy(p)
        size = min(z1.size()[1], z2.size()[1])
        z1 = z1[:, :size]
        z2 = z2[:, :size]
        z = torch.maximum(z1, z2)
        return z
