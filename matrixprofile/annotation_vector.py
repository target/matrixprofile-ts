# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np
from .utils import movmeanstd


def make_complexity_AV(ts, m):
    """
    returns a complexity annotation vector for timeseries ts with window m.
    The complexity of a window is the average absolute difference between consecutive data points.
    """
    diffs = np.diff(ts, append=0)**2
    diff_mean, diff_std = movmeanstd(diffs, m)

    complexity = np.sqrt(diff_mean)
    complexity = complexity - complexity.min()
    complexity = complexity / complexity.max()
    return complexity


def make_meanstd_AV(ts, m):
    """ returns boolean annotation vector which selects windows with a standard deviation greater than average """
    _, std = movmeanstd(ts, m)
    mu = std.mean()
    return (std < mu).astype(int)


def make_clipping_AV(ts, m):
    """
    returns an annotation vector proportional to the number if mins/maxs in the window
    """
    av = (ts == ts.min()) | (ts == ts.max())
    av, _ = movmeanstd(av, m)
    return av
