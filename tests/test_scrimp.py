#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

"""Tests for `scrimp` package."""

import os
import pytest

import numpy as np

import matrixprofile
from matrixprofile import scrimp

MODULE_PATH = matrixprofile.__path__[0]

def test_fast_find_nn_pre():
    """Validate the computations for fast find nn pre."""
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    m = 4

    X, n, sumx2, sumx, meanx, sigmax2, sigmax = scrimp.fast_find_nn_pre(ts, m)
    assert(n == 8)

    expected_sumx2 = np.array([30, 54, 86, 126, 174])
    assert((sumx2 == expected_sumx2)).all()

    expected_sumx = np.array([10, 14, 18, 22, 26])
    assert((sumx == expected_sumx)).all()

    expected_meanx = np.array([2.5, 3.5, 4.5, 5.5, 6.5])
    assert((meanx == expected_meanx).all())

    expected_sigmax2 = np.array([1.25, 1.25, 1.25, 1.25, 1.25])
    assert(np.allclose(sigmax2, expected_sigmax2))

    expected_sigmax = np.array([1.118, 1.118, 1.118, 1.118, 1.118])
    assert(np.allclose(sigmax, expected_sigmax, 1e-02))


def test_calc_exclusion_zone():
    assert(scrimp.calc_exclusion_zone(4) == 1)


def test_calc_step_size():
    assert(scrimp.calc_step_size(4, 0.25) == 1)


def test_calc_profile_len():
    assert(scrimp.calc_profile_len(8, 4) == 5)


def test_time_series_too_short_exception():
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 4, 0.25)
        assert 'Time series is too short' in str(excinfo.value)


def test_window_size_minimum_exception():
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, 0.25)
        assert 'Window size must be at least 4' in str(excinfo.value)


def test_invalid_step_size_negative():
    exc = 'step_size should be a float between 0 and 1.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, -1)
        assert exc in str(excinfo.value)


def test_invalid_step_size_str():
    exc = 'step_size should be a float between 0 and 1.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, 'a')
        assert exc in str(excinfo.value)


def test_invalid_step_size_greater():
    exc = 'step_size should be a float between 0 and 1.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, 2)
        assert exc in str(excinfo.value)


def test_invalid_runtime_str():
    exc = 'runtime should be a valid positive integer.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, runtime='1')
        assert exc in str(excinfo.value)


def test_invalid_runtime_zero():
    exc = 'runtime should be a valid positive integer.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, runtime=0)
        assert exc in str(excinfo.value)


def test_invalid_runtime_negative():
    exc = 'runtime should be a valid positive integer.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, runtime=-1)
        assert exc in str(excinfo.value)


def test_invalid_random_state_exception():
    exc = 'Invalid random_state value given.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, random_state='adsf')
        assert exc in str(excinfo.value)


def test_next_subsequence():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4
    idx = 0

    expected_subsequence = [1, 2, 3, 4]
    assert((scrimp.next_subsequence(ts, idx, m) == expected_subsequence))


def test_calc_distance_profile():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4
    idx = 0

    expected_dp = np.array([
        4.21468485e-08,
        4.21468485e-08,
        0.00000000e+00,
        4.21468485e-08,
        4.21468485e-08
    ])

    subsequence = scrimp.next_subsequence(ts, idx, m)
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = scrimp.fast_find_nn_pre(ts, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)

    np.testing.assert_almost_equal(dp, expected_dp)

    # test idx 1
    idx = 1

    expected_dp = np.array([
        4.21468485e-08,
        4.21468485e-08,
        4.21468485e-08,
        4.21468485e-08,
        4.21468485e-08
    ])

    subsequence = scrimp.next_subsequence(ts, idx, m)
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = scrimp.fast_find_nn_pre(ts, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)

    np.testing.assert_almost_equal(dp, expected_dp)


def test_calc_exclusion_start():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4
    idx = 0
    exclusion_zone = scrimp.calc_exclusion_zone(m)
    start = scrimp.calc_exclusion_start(idx, exclusion_zone)
    expected_start = 0

    assert(expected_start == start)

    idx = 2
    start = scrimp.calc_exclusion_start(idx, exclusion_zone)
    expected_start = 1
    assert(expected_start == start)

    idx = 3
    start = scrimp.calc_exclusion_start(idx, exclusion_zone)
    expected_start = 2
    assert(expected_start == start)


def test_calc_exclusion_stop():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4
    idx = 0
    profile_len = scrimp.calc_profile_len(len(ts), m)  # 4
    exclusion_zone = scrimp.calc_exclusion_zone(m)  # 1
    stop = scrimp.calc_exclusion_stop(idx, exclusion_zone, profile_len)
    expected_stop = 1
    assert(expected_stop == stop)

    idx = 2
    stop = scrimp.calc_exclusion_stop(idx, exclusion_zone, profile_len)
    expected_stop = 3
    assert(expected_stop == stop)

    idx = 3
    stop = scrimp.calc_exclusion_stop(idx, exclusion_zone, profile_len)
    expected_stop = 4
    assert(expected_stop == stop)

    idx = 4
    stop = scrimp.calc_exclusion_stop(idx, exclusion_zone, profile_len)
    expected_stop = 5
    assert(expected_stop == stop)


def test_apply_exclusion_zone():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4

    # test index 0
    idx = 0
    profile_len = scrimp.calc_profile_len(len(ts), m)  # 4
    exclusion_zone = scrimp.calc_exclusion_zone(m)  # 1
    subsequence = scrimp.next_subsequence(ts, idx, m)
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = scrimp.fast_find_nn_pre(ts, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)
    dp = scrimp.apply_exclusion_zone(idx, exclusion_zone, profile_len, dp)

    expected_dp = np.array([
        np.inf,
        np.inf,
        0.4215e-07,
        0.4215e-07,
        0.4215e-07,
    ])

    np.testing.assert_almost_equal(dp, expected_dp)

    # test idx 1
    idx = 1
    subsequence = scrimp.next_subsequence(ts, idx, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)
    dp = scrimp.apply_exclusion_zone(idx, exclusion_zone, profile_len, dp)

    expected_dp = np.array([
        np.inf,
        np.inf,
        np.inf,
        0.4215e-07,
        0.4215e-07,
    ])

    np.testing.assert_almost_equal(dp, expected_dp)

    # test idx 2
    idx = 2
    subsequence = scrimp.next_subsequence(ts, idx, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)
    dp = scrimp.apply_exclusion_zone(idx, exclusion_zone, profile_len, dp)

    expected_dp = np.array([
        0.4215e-07,
        np.inf,
        np.inf,
        np.inf,
        0.4215e-07,
    ])

    np.testing.assert_almost_equal(dp, expected_dp)

    # test idx 3
    idx = 3
    subsequence = scrimp.next_subsequence(ts, idx, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)
    dp = scrimp.apply_exclusion_zone(idx, exclusion_zone, profile_len, dp)

    expected_dp = np.array([
        0.1115e-06,
        0.0421e-06,
        np.inf,
        np.inf,
        np.inf,
    ])

    np.testing.assert_almost_equal(dp, expected_dp)


def test_find_and_store_nn():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4

    # test index 0
    idx = 0
    profile_len = scrimp.calc_profile_len(len(ts), m)
    exclusion_zone = scrimp.calc_exclusion_zone(m)
    subsequence = scrimp.next_subsequence(ts, idx, m)
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = scrimp.fast_find_nn_pre(ts, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)
    dp = scrimp.apply_exclusion_zone(idx, exclusion_zone, profile_len, dp)

    mp = np.zeros(profile_len)
    mp_index = np.zeros(profile_len, dtype='int32')

    mp, mp_index, idx_nn = scrimp.find_and_store_nn(0, idx, mp, mp_index, dp)
    expected_mp = np.array([
        0.4215e-07,
        np.inf,
        0.4215e-07,
        0.4215e-07,
        0.4215e-07,
    ])
    expected_idx_nn = 2
    expected_mp_index = np.array([
        2,
        0,
        0,
        0,
        0,
    ])

    np.testing.assert_almost_equal(mp, expected_mp)
    np.testing.assert_almost_equal(mp_index, expected_mp_index)
    assert(idx_nn == expected_idx_nn)

    # test index 0
    idx = 1
    subsequence = scrimp.next_subsequence(ts, idx, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)
    dp = scrimp.apply_exclusion_zone(idx, exclusion_zone, profile_len, dp)

    mp, mp_index, idx_nn = scrimp.find_and_store_nn(1, idx, mp, mp_index, dp)
    expected_idx_nn = 3
    assert(idx_nn == expected_idx_nn)


def test_calc_idx_diff():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4

    # test index 0
    idx = 0
    profile_len = scrimp.calc_profile_len(len(ts), m)
    exclusion_zone = scrimp.calc_exclusion_zone(m)
    subsequence = scrimp.next_subsequence(ts, idx, m)
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = scrimp.fast_find_nn_pre(ts, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)
    dp = scrimp.apply_exclusion_zone(idx, exclusion_zone, profile_len, dp)

    mp = np.zeros(profile_len)
    mp_index = np.zeros(profile_len, dtype='int32')

    mp, mp_index, idx_nn = scrimp.find_and_store_nn(0, idx, mp, mp_index, dp)

    idx_diff = scrimp.calc_idx_diff(idx, idx_nn)
    expected_idx_diff = 2

    assert(idx_diff == expected_idx_diff)


def test_calc_dotproduct_idx():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4

    # test index 0
    idx = 0
    profile_len = scrimp.calc_profile_len(len(ts), m)
    exclusion_zone = scrimp.calc_exclusion_zone(m)
    subsequence = scrimp.next_subsequence(ts, idx, m)
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = scrimp.fast_find_nn_pre(ts, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)
    dp = scrimp.apply_exclusion_zone(idx, exclusion_zone, profile_len, dp)
    mp = np.zeros(profile_len)
    mp_index = np.zeros(profile_len, dtype='int32')
    mp, mp_index, idx_nn = scrimp.find_and_store_nn(0, idx, mp, mp_index, dp)
    dotproduct = np.zeros(profile_len)
    val = scrimp.calc_dotproduct_idx(dotproduct, m, mp, idx,
                                     sigmax, idx_nn, meanx)
    expected_val = np.array([50, 0, 0, 0, 0])

    np.testing.assert_almost_equal(val, expected_val)


def test_calc_end_idx():
    end_idx = scrimp.calc_end_idx(5, 0, 1, 2)
    expecected_idx = 0

    assert(end_idx == expecected_idx)

    end_idx = scrimp.calc_end_idx(5, 1, 1, 2)
    expecected_idx = 1

    assert(end_idx == expecected_idx)

    end_idx = scrimp.calc_end_idx(5, 2, 1, 2)
    expecected_idx = 2

    assert(end_idx == expecected_idx)


def test_calc_dotproduct_end_idx():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    m = 4

    # test index 0
    idx = 0
    dp = np.array([50, 0, 0, 0, 0])
    endidx = 0
    idx_nn = 2
    idx_diff = 2
    val = scrimp.calc_dotproduct_end_idx(ts, dp, idx, m, endidx, idx_nn, 
                                         idx_diff)
    expected_val = dp

    np.testing.assert_almost_equal(val, expected_val)


def test_calc_refine_distance_end_idx():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4
    step_size = 0.25

    # test index 0
    idx = 0
    profile_len = scrimp.calc_profile_len(len(ts), m)
    exclusion_zone = scrimp.calc_exclusion_zone(m)
    subsequence = scrimp.next_subsequence(ts, idx, m)
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = scrimp.fast_find_nn_pre(ts, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)
    dp = scrimp.apply_exclusion_zone(idx, exclusion_zone, profile_len, dp)
    mp = np.zeros(profile_len)
    mp_index = np.zeros(profile_len, dtype='int32')
    mp, mp_index, idx_nn = scrimp.find_and_store_nn(0, idx, mp, mp_index, dp)
    idx_diff = scrimp.calc_idx_diff(idx, idx_nn)
    step_size = scrimp.calc_step_size(m, step_size)
    endidx = scrimp.calc_end_idx(profile_len, idx, step_size, idx_diff)
    refine_distance = np.full(profile_len, np.inf)
    result = scrimp.calc_refine_distance_end_idx(
        refine_distance, dp, idx, endidx, meanx, sigmax, idx_nn, idx_diff, m)
    expected_result = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])

    np.testing.assert_almost_equal(result, expected_result)


def test_calc_begin_idx():
    step_size = 1
    idx_diff = 2

    idx = 0
    val = scrimp.calc_begin_idx(idx, step_size, idx_diff)
    assert(val == 0)

    idx = 1
    val = scrimp.calc_begin_idx(idx, step_size, idx_diff)
    assert(val == 1)


def test_calc_dotproduct_begin_idx():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4

    # test index 0
    idx = 0
    dp = np.array([50, 0, 0, 0, 0])
    beginidx = 0
    idx_nn = 2
    idx_diff = 2
    val = scrimp.calc_dotproduct_begin_idx(
        ts, dp, beginidx, idx, idx_diff, m, idx_nn)
    expected_val = np.array([50, 0, 0, 0, 0])

    np.testing.assert_almost_equal(val, expected_val)


def test_calc_refine_distance_begin_idx():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    m = 4
    step_size = 0.25

    # test index 0
    idx = 0
    profile_len = scrimp.calc_profile_len(len(ts), m)
    exclusion_zone = scrimp.calc_exclusion_zone(m)
    subsequence = scrimp.next_subsequence(ts, idx, m)
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = scrimp.fast_find_nn_pre(ts, m)
    dp = scrimp.calc_distance_profile(X, subsequence, n, m, meanx, sigmax)
    dp = scrimp.apply_exclusion_zone(idx, exclusion_zone, profile_len, dp)
    mp = np.zeros(profile_len)
    mp_index = np.zeros(profile_len, dtype='int32')
    mp, mp_index, idx_nn = scrimp.find_and_store_nn(0, idx, mp, mp_index, dp)
    idx_diff = scrimp.calc_idx_diff(idx, idx_nn)
    step_size = scrimp.calc_step_size(m, step_size)
    beginidx = scrimp.calc_begin_idx(idx, step_size, idx_diff)

    refine_distance = np.full(profile_len, np.inf)
    result = scrimp.calc_refine_distance_begin_idx(
        refine_distance, dp, beginidx, idx, idx_diff, idx_nn, sigmax, meanx, m)
    expected_result = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])

    np.testing.assert_almost_equal(result, expected_result)


def test_calc_curlastz():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    m = 4
    n = len(ts)
    profile_len = scrimp.calc_profile_len(n, m)
    
    # test index 2
    idx = 2
    curlastz = np.zeros(profile_len)
    curlastz = scrimp.calc_curlastz(ts, m, n, idx, profile_len, curlastz)
    expected_result = np.array([0, 0, 50, 82, 122])

    np.testing.assert_almost_equal(curlastz, expected_result)

    # test index 3
    idx = 3
    curlastz = np.zeros(profile_len)
    curlastz = scrimp.calc_curlastz(ts, m, n, idx, profile_len, curlastz)
    expected_result = np.array([0, 0, 0, 60, 96])

    np.testing.assert_almost_equal(curlastz, expected_result)


def test_calc_curdistance():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    m = 4

    # test index 2
    idx = 2
    profile_len = scrimp.calc_profile_len(len(ts), m)
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = scrimp.fast_find_nn_pre(ts, m)
    curlastz = np.zeros(profile_len)
    curlastz = scrimp.calc_curlastz(ts, m, n, idx, profile_len, curlastz)

    curdistance = np.zeros(profile_len)
    curdistance = scrimp.calc_curdistance(curlastz, meanx, sigmax, idx, 
                                          profile_len, m, curdistance)

    expected_result = np.array([
        0,
        0,
        0.4215e-07,
        0.4215e-07,
        0.4215e-07,
    ])

    np.testing.assert_almost_equal(curdistance, expected_result)

    # test index 3
    idx = 3
    curlastz = np.zeros(profile_len)
    curlastz = scrimp.calc_curlastz(ts, m, n, idx, profile_len, curlastz)
    curdistance = np.zeros(profile_len)
    curdistance = scrimp.calc_curdistance(curlastz, meanx, sigmax, idx, 
                                          profile_len, m, curdistance)

    np.testing.assert_almost_equal(curdistance, expected_result)


def test_scrimp_plus_plus():
    ts = np.array([0, 0, 1, 0, 0, 0, 1, 0])
    m = 4
    step_size = 0.25
    mp, mpidx = scrimp.scrimp_plus_plus(ts, m, step_size)

    expected_mpidx = np.array([
        4,
        3,
        0,
        0,
        0,
    ])

    np.testing.assert_almost_equal(mpidx, expected_mpidx)

    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    m = 32
    step_size = 0.25
    mp, mpidx = scrimp.scrimp_plus_plus(ts, m, step_size)
    expected_mp = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'mp.txt'))

    np.testing.assert_almost_equal(mp, expected_mp, decimal=4)


def test_runtime_exceeded_warns():
    ts = np.arange(2**18)
    m = 2**8
    runtime = 1

    warn_text = 'Max runtime exceeded. Approximate solution is given.'
    with pytest.warns(RuntimeWarning, match=warn_text):
        scrimp.scrimp_plus_plus(ts, m, runtime=runtime)