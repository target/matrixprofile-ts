import sys
from matrixprofile.discords import *
import numpy as np
import pytest

class TestClass(object):

    def test_discords_no_exclusion(self):
        mp = np.array([1.0, 2.0, 3.0, 4.0])
        outcome = np.array([3,3,3,3])

        assert(np.allclose(discords(mp, 0, 4), outcome))

    def test_discords_exclude_one(self):
        mp = np.array([1.0, 2.0, 3.0, 4.0])
        outcome = np.array([3,1,sys.maxsize,sys.maxsize])

        assert(np.allclose(discords(mp, 1, 4), outcome))

    def test_discords_exclude_big(self):
        mp = np.array([1.0, 2.0, 3.0, 4.0])
        outcome = np.array([3,sys.maxsize,sys.maxsize,sys.maxsize])

        assert(np.allclose(discords(mp, 10, 4), outcome))

    def test_discords_empty_mp(self):
        mp = np.array([])
        outcome = np.array([])

        assert(np.allclose(discords(mp, 1, 4), outcome))

    def test_discords_k_larger_than_mp(self):
        mp = np.array([1.0, 2.0, 3.0, 4.0])
        outcome = np.array([3,1,sys.maxsize,sys.maxsize])

        assert(np.allclose(discords(mp, 1, 10), outcome))

