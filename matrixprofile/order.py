# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import random

class Order:
    '''These objects define the order in which the distance profiles are calculated for a given matrix profile'''
    def next(self):
        raise NotImplementedError("next() not implemented")

class linearOrder(Order):
    def __init__(self,m):
        self.m = m
        self.idx = -1

    def next(self):
        self.idx += 1
        if self.idx < self.m:
            return self.idx
        else:
            return None


class randomOrder(Order):
    def __init__(self,m):
        self.idx = -1
        self.indices = list(range(m))
        random.shuffle(self.indices)

    def next(self):
        self.idx += 1
        try:
            return self.indices[self.idx]

        except IndexError:
            return None
