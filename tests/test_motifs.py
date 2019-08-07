import sys
from matrixprofile.motifs import *
import numpy as np
import pytest


class TestClass(object):

    def test_motifs_ind(self):
        a = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                      1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        mp = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        mpi = np.array([4., 5., 6., 7., 0., 1., 2., 3., 0.])

        motif, _ = motifs(a, (mp, mpi))
        motif_outcome = [[0, 4, 8]]

        assert(motif == motif_outcome)

    def test_motifs_dist(self):
        a = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                      1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        mp = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        mpi = np.array([4., 5., 6., 7., 0., 1., 2., 3., 0.])

        _, dist = motifs(a, (mp, mpi))
        motif_dist = [0]

        assert(np.allclose(dist, motif_dist))

    def test_motifs_noexclude(self):
        a = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                      1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        mp = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        mpi = np.array([4., 5., 6., 7., 0., 1., 2., 3., 0.])

        motif, _ = motifs(a, (mp, mpi), ex_zone=0)
        motif_outcome = [[0, 4, 8], [1, 5], [2, 6]]
        assert(motif == motif_outcome)

    def test_motifs_more(self):
        a = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                      1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        mp = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        mpi = np.array([4., 5., 6., 7., 0., 1., 2., 3., 0.])

        motif, _ = motifs(a, (mp, mpi), max_motifs=5, ex_zone=0)
        motif_outcome = [[0, 4, 8], [1, 5], [2, 6], [3, 7]]

        assert(motif == motif_outcome)

    def test_motifs_nneighbors(self):
        a = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                      1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        mp = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        mpi = np.array([4., 5., 6., 7., 0., 1., 2., 3., 0.])

        motif, _ = motifs(a, (mp, mpi), n_neighbors=2)
        motif_outcome = [[0, 4]]

        assert(motif == motif_outcome)

    def test_motifs_empty(self):
        res = motifs([], ([], []))
        assert(res == ([], []))

    def test_motifs_len_check(self):
        with pytest.raises(ValueError) as excinfo:
            motifs([1, 2], ([1, 2], [1, 2]))
        assert 'Matrix profile is longer than time series.' in str(
            excinfo.value)
