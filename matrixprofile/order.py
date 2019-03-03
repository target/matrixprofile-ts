# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

class Order:
    """
    An object that defines the order in which the distance profiles are calculated for a given Matrix Profile
    """
    def next(self):
        raise NotImplementedError("next() not implemented")

class linearOrder(Order):
    """
    An object that defines a linear (iterative) order in which the distance profiles are calculated for a given Matrix Profile
    """
    def __init__(self,m):
        self.m = m
        self.idx = -1

    def next(self):
        """
        Advances the Order object to the next index
        """
        self.idx += 1
        if self.idx < self.m:
            return self.idx
        else:
            return None


class randomOrder(Order):
    """
    An object that defines a random order in which the distance profiles are calculated for a given Matrix Profile
    """
    def __init__(self,m, random_state=None):
        self.idx = -1
        self.indices = np.arange(m)
        self.random_state = random_state
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        np.random.shuffle(self.indices)

    def next(self):
        """
        Advances the Order object to the next index
        """
        self.idx += 1
        try:
            return self.indices[self.idx]

        except IndexError:
            return None
