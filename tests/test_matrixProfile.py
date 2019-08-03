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
        mpi_outcome = np.array([0., 1., 2., 3., 0., 1., 2., 3., 0.])

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
        mpi_outcome = np.array([0., 1., 2., 3., 0., 1., 2., 3., 0.])

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


        r = stamp(a,4, sampling=1.0)

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

    def test_stamp_dual_mp_nan_inf(self):
        a = np.array([0.0,1.0,1.0,0.0,np.nan,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.0,np.inf,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
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
        r = stamp(a,4, sampling=1.0)
        final = np.round(stampi_update(a,4,r[0],r[1],95),2)

        mp_outcome = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.83])

        assert(np.allclose(final[0],mp_outcome))


    # def test_stampi_mpi(self):
    #     #Note: given the new self-join logic in v0.0.7 and above, STAMPI will not guarantee the same MPI where there are multiple "true" outcomes. There are 128 possible outcomes, so this test needs to be better designed...
    #     a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0])
    #     r = stamp(a,4, sampling=1.0)
    #     final = np.round(stampi_update(a,4,r[0],r[1],95),2)
    #
    #     mpi_outcome_1 = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 3.0])
    #     mpi_outcome_2 = np.array([4.0, 1.0, 6.0, 7.0, 4.0, 1.0, 6.0, 7.0, 3.0])
    #     mpi_outcome_3 = np.array([4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 3.0])
    #     mpi_outcome_4 = np.array([0.0, 5.0, 6.0, 7.0, 0.0, 5.0, 6.0, 7.0, 3.0])
    #     ...
    #
    #
    #
    #     assert(np.allclose(final[1],mpi_outcome_1) | np.allclose(final[1],mpi_outcome_2) | np.allclose(final[1],mpi_outcome_3) | np.allclose(final[1],mpi_outcome_4))


    def test_stomp_self_mp(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mp_outcome = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])


        r = stomp(a,4)

        assert(r[0] == mp_outcome).all()


    def test_stomp_self_mpi(self):
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        mpi_outcome = np.array([4., 5., 6., 7., 0., 1., 2., 3., 0.])

        r = stomp(a,4)

        assert(r[1] == mpi_outcome).all()

    def test_stamp_sampling_over_one(self):
        with pytest.raises(ValueError) as excinfo:
            stamp(None,None,sampling=2)
        assert 'Sampling value must be a percentage' in str(excinfo.value)

    def test_stamp_sampling_under_zero(self):
        with pytest.raises(ValueError) as excinfo:
            stamp(None,None,sampling=-1)
        assert 'Sampling value must be a percentage' in str(excinfo.value)

    def test_stamp_random_state_same_results_self_join(self):
        random_state = 99
        sampling = 0.30
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])

        r = stamp(a,4,None,sampling=sampling,random_state=random_state)
        r2 = stamp(a,4,None,sampling=sampling,random_state=random_state)

        all_same = (r[0] == r2[0]).all() and (r[1] == r2[1]).all()
        assert(all_same == True)


    def test_stamp_random_state_same_results_dual_join(self):
        random_state = 99
        sampling = 0.30
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.2,0.0,1.0,1.0,0.3,0.0,1.0,1.0,0.0])

        r = stamp(a,4,b,sampling=sampling,random_state=random_state)
        r2 = stamp(a,4,b,sampling=sampling,random_state=random_state)

        all_same = (r[0] == r2[0]).all() and (r[1] == r2[1]).all()
        assert(all_same == True)

    def test_stamp_with_parallel_version_random_state_set_self_join(self):
        random_state = 99
        sampling = 0.1
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        r = stamp(a,4,None,sampling=sampling,random_state=random_state)
        r2 = stamp(a,4,None,sampling=sampling,random_state=random_state)

        all_same = (r[0] == r2[0]).all() and (r[1] == r2[1]).all()
        assert(all_same == True)

    def test_stamp_with_parallel_version_random_state_set_dual_join(self):
        random_state = 99
        sampling = 0.1
        a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,2.0,0.0,1.1,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
        r = stamp(a,4,b,sampling=sampling,random_state=random_state)
        r2 = stamp(a,4,b,sampling=sampling,random_state=random_state)

        all_same = (r[0] == r2[0]).all() and (r[1] == r2[1]).all()
        assert(all_same == True)
