# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

def extract_regimes(cac, window_size, num=3):
    """
    Given a corrected arc curve extract a given number of regimes. The regimes
    are computed with an exclusion zone to avoid very close matches. By default
    it sets the exclusion zone to 5 times the window size per the author.

    Per the authors:

    This exclusion zone is based on an assumption that regimes will have
    multiple repetitions; FLUSS is not able to segment single gesture 
    patterns.

    Parameters
    ----------
    cac: Array like data structure representing the corrected arc curve.
    window_size: The window size used to compute the matrix profile.
    num: The number of regimes to return.

    Returns
    -------
    Array of index locations for the starting point of the regimes.
    """
    ez = window_size * 5
    found = []
    tmp = np.copy(cac)
    n = len(tmp)
    
    for _ in range(num):
        min_index = np.argmin(tmp)
        found.append(min_index)
        
        # apply exclusion zone
        ez_start = np.max([0, min_index - ez])
        ez_end = np.min([n, min_index + ez])
        tmp[ez_start:ez_end] = np.inf

    return np.array(found, dtype=int)