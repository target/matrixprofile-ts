from matrixprofile.fluss import *
import numpy as np

class TestClass(object):

    def test_fluss(self):
        cac = fluss([3, 0, 1, 1, 1, 7, 3, 8, 6, 7, 8, 7], 2)
        outcome = np.array([1., 1., 0.9, 0.22222222, 0.1875, 0.17142857, 0.16666667, 0.17142857, 0.375, 0.44444444,
                            1., 1.])

        assert(np.allclose(cac, outcome))
