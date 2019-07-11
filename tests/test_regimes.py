import os

from matrixprofile.matrixProfile import stomp
from matrixprofile.fluss import fluss
from matrixprofile.regimes import extract_regimes

import matrixprofile

MODULE_PATH = matrixprofile.__path__[0]

import numpy as np

class TestClass(object):

    def test_extract_regimes(self):
        data_file = os.path.join(MODULE_PATH, '..', 'docs', 'examples', 'rawdata.csv')
        ts = np.loadtxt(data_file, skiprows=1)
        m = 32
        mp, pi = stomp(ts, m)

        cac = fluss(pi, m)
        
        # test with 3 regimes
        regimes = extract_regimes(cac, m)
        expected = np.array([759, 423, 583])

        np.testing.assert_array_equal(regimes, expected)

        # test with 2 regimes
        regimes = extract_regimes(cac, m, num=2)
        expected = np.array([759, 423])

        np.testing.assert_array_equal(regimes, expected)