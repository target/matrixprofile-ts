# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np
from . import distanceProfile

def motifs(ts, mp, m, k=3, radius=2, n_neighbors=None, ex_zone=None):
    """
    Computes the top k motifs from a matrix profile

    Parameters
    ----------
    mp: tuple, (matrix profile numpy array, matrix profile indices)
    m: window size (should match window used to calculate mp)
    k: the number of motifs to discover
    ex_zone: the number of samples to exclude and set to Inf on either side of a found motifs
        defaults to m/2

    Returns a list of sets of indexes representing the motif starting locations.
    """
    if ex_zone is None:
        ex_zone = m/2

    motifs = []
    mp_current = np.copy(mp[0])

    for j in range(k):
        # find minimum distance and index location
        min_idx = mp_current.argmin()
        motif_distance = mp_current[min_idx]
        if motif_distance == np.inf:
            return motifs

        # filter out all indexes that have a distance within r*motif_distance
        motif_set = set()
        initial_motif = [min_idx, int(mp[1][min_idx])]
        motif_set = set(initial_motif)

        prof, _ = distanceProfile.massDistanceProfile(ts, initial_motif[0], m)

        # kill off any indices around the initial motif pair since they are
        # trivial solutions
        _applyExclusionZone(prof, initial_motif[0], ex_zone)
        _applyExclusionZone(prof, initial_motif[1], ex_zone)
        # exclude previous motifs
        for ms, _ in motifs:
            for idx in ms:
                _applyExclusionZone(prof, idx, ex_zone)

        # keep looking for the closest index to the current motif. Each
        # index found will have an exclusion zone applied as to remove
        # trivial solutions. This eventually exits when there's nothing
        # found within the radius distance.
        prof_idx_sort = prof.argsort()

        for nn_idx in prof_idx_sort:
            if prof[nn_idx] < motif_distance*radius:
                motif_set.add(nn_idx)
                _applyExclusionZone(prof, nn_idx, ex_zone)
                if len(motif_set) == n_neighbors:
                    break
            else:
                break

        motifs += [(list(sorted(motif_set)), motif_distance)]
        for motif in motif_set:
            _applyExclusionZone(mp_current, motif, ex_zone)

    return motifs


def _applyExclusionZone(prof, idx, zone):
    start = int(max(0, idx-zone))
    end = int(idx + zone + 1)
    prof[start:end] = np.inf
