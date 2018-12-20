from matrixprofile.utils import *
import numpy as np
import pytest

class TestClass(object):

    def test_zNormalize_zero_error(self):
        with pytest.raises(ValueError):
            zNormalize(np.array([2.0,2.0,2.0]))

    def test_zNormalize(self):
        outcome = np.array([-1.0,1.0,1.0,-1.0])
        assert np.allclose(zNormalize(np.array([0.0,1.0,1.0,0.0])),outcome)

    def test_zNormalizeEuclidian_length_error(self):
        with pytest.raises(ValueError):
            zNormalizeEuclidian(np.array([1,2,3]),np.array([1,2]))

    def test_zNormalizeEuclidian(self):
        a = np.array([0.0,1.0,1.0,0.0])
        b = np.array([1.0,2.0,1.0,2.0])

        assert np.round(zNormalizeEuclidian(a,b),4) == np.round(2.0*np.sqrt(2.0),4)

    def test_movstd(self):
        a = np.array([1.0,2.0,4.0,8.0])
        m = 2

        assert np.allclose(movstd(a,m), np.array([0.5,1.0,2.0]))

    def test_slidingDotProduct(self):
        a = np.array([1.0,2.0])
        b = np.array([1.0,2.0,3.0,4.0])

        outcome = np.array([5.0,8.0,11.0])

        assert np.allclose(slidingDotProduct(a,b), outcome)

    def test_slidingDotProduct_odd_ts(self): #Good
        a = np.array([1.0,2.0])
        b = np.array([1.0,2.0,3.0])

        outcome = np.array([5.0,8.0])

        assert np.allclose(slidingDotProduct(a,b), outcome)

    def test_slidingDotProduct_odd_query(self):
        a = np.array([1.0,2.0,3.0])
        b = np.array([1.0,2.0,3.0,4.0])

        outcome = np.array([14.0,20.0])

        assert np.allclose(slidingDotProduct(a,b), outcome)

    def test_slidingDotProduct_odd_both(self):

        #Something weird going on with this test...the numbers are correct but the vectors aren't showing up as equal...
        a = np.array([1.0,2.0,3.0])
        b = np.array([1.0,2.0,3.0,4.0,5.0])

        outcome = np.array([14.,20.,26.])

        assert np.allclose(slidingDotProduct(a,b), outcome)

    def test_mass(self):
        a = np.array([0.0,1.0,1.0,0.0])
        b = np.array([0.0,4.0,4.0,0.0])

        outcome = np.array([0.0])

        assert np.allclose(mass(a,b), outcome)


    def test_apply_av(self):
        a = [np.array([1.0,2.0,1.0,2.0]),np.array([0.0,0.0,0.0,0.0])]
        av = np.array([2.0,1.0,2.0,1.0])

        outcome = np.array([2.0,2.0,2.0,2.0])

        assert np.allclose(apply_av(a,av), outcome)


    def test_apply_av_length_error(self):
        a = [np.array([1.0,2.0,1.0,2.0]),np.array([0.0,0.0,0.0,0.0])]
        av = np.array([2.0,1.0,2.0])

        with pytest.raises(ValueError):
            apply_av(a,av)
