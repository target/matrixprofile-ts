from matrixprofile.matrixProfile import *
import numpy as np
import pytest

class TestClass(object):

    def test_naiveMP_self_mp(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mp_outcome = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])


        r = naiveMP(a,4)

        assert(r[0] == mp_outcome).all()


    def test_naiveMP_self_mpi(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mpi_outcome = np.array([4., 5., 6., 7., 0., 1., 2., 3., 0.])

        r = naiveMP(a,4)

        assert(r[1] == mpi_outcome).all()

    def test_naiveMP_dual_mp(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mp_outcome = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])


        r = naiveMP(a,4,b)

        assert(r[0] == mp_outcome).all()


    def test_naiveMP_dual_mpi(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mpi_outcome = np.array([0., 1., 2., 3., 0., 1., 2., 3., 0.])

        r = naiveMP(a,4,b)

        assert(r[1] == mpi_outcome).all()

    def test_stmp_self_mp(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mp_outcome = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])


        r = stmp(a,4)

        assert(r[0] == mp_outcome).all()


    def test_stmp_self_mpi(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mpi_outcome = np.array([4., 5., 6., 7., 0., 1., 2., 3., 0.])

        r = stmp(a,4)

        assert(r[1] == mpi_outcome).all()

    def test_stmp_dual_mp(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mp_outcome = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])


        r = stmp(a,4,b)

        assert(r[0] == mp_outcome).all()


    def test_stmp_dual_mpi(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mpi_outcome = np.array([0., 1., 2., 3., 0., 1., 2., 3., 0.])

        r = stmp(a,4,b)

        assert(r[1] == mpi_outcome).all()


    def test_stamp_self_mp(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mp_outcome = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])


        r = stamp(a,4)

        assert(r[0] == mp_outcome).all()


    def test_stamp_self_mpi(self):
        #Note that we're only testing for the length of the matrix profile index and not the specific values
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mpi_outcome = np.array([4., 5., 6., 7., 0., 1., 2., 3., 0.])

        r = stamp(a,4,sampling=1.0)

        assert(len(r[1]) == 9)

    def test_stamp_dual_mp(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mp_outcome = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])


        r = stamp(a,4,b,sampling=1.0)

        assert(r[0] == mp_outcome).all()


    def test_stamp_dual_mpi(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mpi_outcome = np.array([0., 1., 2., 3., 0., 1., 2., 3., 0.])

        r = stamp(a,4,b,sampling=1.0)

        assert(len(r[1]) == 9)


    def test_stampi_mp(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0])
        r = stamp(a,4)
        final = np.round(stampi_update(a,4,r[0],r[1],95),2)

        mp_outcome = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.83])

        assert(np.allclose(final[0],mp_outcome))


    def test_stampi_mpi(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0])
        r = stamp(a,4)
        final = np.round(stampi_update(a,4,r[0],r[1],95),2)

        mpi_outcome = np.array([4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 3.0])

        assert(np.allclose(final[1],mpi_outcome))
