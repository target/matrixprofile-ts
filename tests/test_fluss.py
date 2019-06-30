from matrixprofile.fluss import *
import numpy as np

class TestClass(object):

    def test_fluss(self):
        cac = fluss([3, 1, 2, 1, 1, 7, 3, 7, 6, 7, 8, 7], 2)
        outcome = np.array([1., 1., 0.9, 0.44444444, 0.1875, 0.34285714, 0.33333333, 0.51428571, 0.5625, 0.44444444, 1.,
                            1.])

        assert(np.allclose(cac, outcome))