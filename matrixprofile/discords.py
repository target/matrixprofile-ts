# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import sys
import numpy as np

def discords(mp,ex_zone,k=3):
    """
    Computes the top k discords from a matrix profile

    Parameters
    ----------
    mp: matrix profile numpy array
    k: the number of discords to discover
    ex_zone: the number of samples to exclude and set to Inf on either side of a found discord

    Returns a list of indexes represent the discord starting locations. MaxInt indicates there
    were no more discords that could be found due to too many exclusions or profile being too
    small. Discord start indices are sorted by highest matrix profile value.
    """
    k = len(mp) if k > len(mp) else k

    mp_current = np.copy(mp)
    d = np.zeros(k, dtype='int')
    for i in range(k):
        maxVal = 0
        maxIdx = sys.maxsize
        for j, val in enumerate(mp_current):
            if not np.isinf(val) and val > maxVal:
                maxVal = val
                maxIdx = j

        d[i] = maxIdx
        mp_current[max([maxIdx-ex_zone, 0]):min([maxIdx+ex_zone, len(mp_current)])] = np.inf

    return d
