# -*- coding: utf-8 -*-

"""
This module consists of all code to implement the SCRIMP++ algorithm. SCRIMP++ 
is an anytime algorithm that computes the matrix profile for a given time 
series (ts) over a given window size (m).

This algorithm was originally created at the University of California 
Riverside. For further academic understanding, please review this paper:

Matrix Proﬁle XI: SCRIMP++: Time Series Motif Discovery at Interactive
Speed. Yan Zhu, Chin-Chia Michael Yeh, Zachary Zimmerman, Kaveh Kamgar
Eamonn Keogh, ICDM 2018.

https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math
import time

import warnings

import numpy as np


def fast_find_nn_pre(ts, m):
    n = len(ts)
    X = np.fft.fft(ts)
    cum_sumx = np.cumsum(ts)
    cum_sumx2 = np.cumsum(np.power(ts, 2))
    sumx = cum_sumx[m-1:n] - np.insert(cum_sumx[0:n-m], 0, 0)
    sumx2 = cum_sumx2[m-1:n] - np.insert(cum_sumx2[0:n-m], 0, 0)
    meanx = sumx / m
    sigmax2 = (sumx2 / m) - np.power(meanx, 2)
    sigmax = np.sqrt(sigmax2)

    return (X, n, sumx2, sumx, meanx, sigmax2, sigmax)


def calc_distance_profile(X, y, n, m, meanx, sigmax):
    # reverse the query
    y = np.flip(y, 0)
   
    # make y same size as ts with zero fill
    y = np.concatenate([y, np.zeros(n-m)])

    # main trick of getting dot product in O(n log n) time
    Y = np.fft.fft(y)
    Z = X * Y
    z = np.fft.ifft(Z)

    # compute y stats in O(n)
    sumy = np.sum(y)
    sumy2 = np.sum(np.power(y, 2))
    meany = sumy / m
    sigmay2 = sumy2 / m - meany ** 2
    sigmay = np.sqrt(sigmay2)

    dist = (z[m - 1:n] - m * meanx * meany) / (sigmax * sigmay)
    dist = m - dist
    dist = np.real(2 * dist)

    return np.absolute(np.sqrt(dist))


def calc_exclusion_zone(window_size):
    return window_size / 4


def calc_step_size(window_size, step_size):
    return int(math.floor(window_size * step_size))


def calc_profile_len(n, window_size):
    return n - window_size + 1


def next_subsequence(ts, idx, m):
    return ts[idx:idx + m]


def calc_exclusion_start(idx, exclusion_zone):
    return int(np.max([0, idx - exclusion_zone]))


def calc_exclusion_stop(idx, exclusion_zone, profile_len):
    return int(np.min([profile_len, idx + exclusion_zone]))


def apply_exclusion_zone(idx, exclusion_zone, profile_len, distance_profile):
    exc_start = calc_exclusion_start(idx, exclusion_zone)
    exc_stop = calc_exclusion_stop(idx, exclusion_zone, profile_len)
    distance_profile[exc_start:exc_stop + 1] = np.inf

    return distance_profile


def find_and_store_nn(iteration, idx, matrix_profile, mp_index, 
                      distance_profile):
    if iteration == 0:
        matrix_profile = distance_profile
        mp_index[:] = idx
    else:
        update_pos = distance_profile < matrix_profile
        mp_index[update_pos] = idx
        matrix_profile[update_pos] = distance_profile[update_pos]

    idx_min = np.argmin(distance_profile)
    matrix_profile[idx] = distance_profile[idx_min]
    mp_index[idx] = idx_min
    idx_nn = mp_index[idx]

    return (matrix_profile, mp_index, idx_nn)


def calc_idx_diff(idx, idx_nn):
    return idx_nn - idx


def calc_dotproduct_idx(dotproduct, m, mp, idx, sigmax, idx_nn, meanx):
    dotproduct[idx] = (m - mp[idx] ** 2 / 2) * \
        sigmax[idx] * sigmax[idx_nn] + m * meanx[idx] * meanx[idx_nn]

    return dotproduct


def calc_end_idx(profile_len, idx, step_size, idx_diff):
    return np.min([profile_len - 1, idx + step_size - 1, 
                  profile_len - idx_diff])


def calc_dotproduct_end_idx(ts, dp, idx, m, endidx, idx_nn, idx_diff):
    tmp_a = ts[idx+m:endidx+m]
    tmp_b = ts[idx_nn+m:endidx+m+idx_diff]
    tmp_c = ts[idx:endidx]
    tmp_d = ts[idx_nn:endidx+idx_diff]
    tmp_f = tmp_a * tmp_b - tmp_c * tmp_d

    dp[idx+1:endidx+1] = dp[idx] + np.cumsum(tmp_f)

    return dp


def calc_refine_distance_end_idx(refine_distance, dp, idx, endidx, meanx, 
                                 sigmax, idx_nn, idx_diff, m):
    tmp_a = dp[idx+1:endidx+1]
    tmp_b = meanx[idx+1:endidx+1]
    tmp_c = meanx[idx_nn+1:endidx+idx_diff+1]
    tmp_d = sigmax[idx+1:endidx+1]
    tmp_e = sigmax[idx_nn+1:endidx+idx_diff+1]
    tmp_f = tmp_b * tmp_c
    tmp_g = tmp_d * tmp_e
    tmp_h = (m-(tmp_a - m * tmp_f) / (tmp_g))
    refine_distance[idx+1:endidx+1] = np.sqrt(np.abs(2 * tmp_h))    

    return refine_distance


def calc_begin_idx(idx, step_size, idx_diff):
    return np.max([0, idx - step_size + 1, 2 - idx_diff])


def calc_dotproduct_begin_idx(ts, dp, beginidx, idx, idx_diff, m, 
                              idx_nn):
    indices = list(range(idx - 1, beginidx - 1, -1))    

    if not indices:
        return dp

    tmp_a = ts[indices]
    indices_b = list(range(idx_nn - 1, beginidx + idx_diff - 1, -1))
    tmp_b = ts[indices_b]
    indices_c = list(range(idx + m - 1, beginidx + m - 1, -1))
    tmp_c = ts[indices_c]
    indices_d = list(range(idx_nn - 1 + m, beginidx + idx_diff + m - 1, -1))
    tmp_d = ts[indices_d]

    dp[indices] = dp[idx] + \
        np.cumsum((tmp_a * tmp_b) - (tmp_c * tmp_d))

    return dp


def calc_refine_distance_begin_idx(refine_distance, dp, beginidx, idx, 
                                   idx_diff, idx_nn, sigmax, meanx, m):
    if not (beginidx < idx):
        return refine_distance

    tmp_a = dp[beginidx:idx]
    tmp_b = meanx[beginidx:idx]
    tmp_c = meanx[beginidx+idx_diff:idx_nn]
    tmp_d = sigmax[beginidx:idx]
    tmp_e = sigmax[beginidx+idx_diff:idx_nn]
    tmp_f = tmp_b * tmp_c
    tmp_g = tmp_d * tmp_e
    tmp_h = (m-(tmp_a - m * tmp_f) / (tmp_g))

    refine_distance[beginidx:idx] = np.sqrt(np.abs(2 * tmp_h))

    return refine_distance


def apply_update_positions(matrix_profile, mp_index, refine_distance, beginidx,
                           endidx, orig_index, idx_diff):
    tmp_a = refine_distance[beginidx:endidx+1]
    tmp_b = matrix_profile[beginidx:endidx+1]
    update_pos1 = np.argwhere(tmp_a < tmp_b).flatten()    

    if len(update_pos1) > 0:        
        update_pos1 = update_pos1 + beginidx
        matrix_profile[update_pos1] = refine_distance[update_pos1]
        mp_index[update_pos1] = orig_index[update_pos1] + idx_diff

    tmp_a = refine_distance[beginidx:endidx + 1]
    tmp_b = matrix_profile[beginidx + idx_diff:endidx + idx_diff + 1]
    update_pos2 = np.argwhere(tmp_a < tmp_b).flatten()

    if len(update_pos2) > 0:
        update_pos2 = update_pos2 + beginidx
        matrix_profile[update_pos2 + idx_diff] = refine_distance[update_pos2]
        mp_index[update_pos2 + idx_diff] = orig_index[update_pos2] - idx_diff

    return (matrix_profile, mp_index)


def calc_curlastz(ts, m, n, idx, profile_len, curlastz):
    curlastz[idx] = np.sum(ts[0:m] * ts[idx:idx+m])

    tmp_a = ts[m:n - idx]
    tmp_b = ts[idx + m:n]
    tmp_c = ts[0:profile_len - idx - 1]
    tmp_d = ts[idx:profile_len - 1]
    tmp_e = tmp_a * tmp_b
    tmp_f = tmp_c * tmp_d
    curlastz[idx+1:profile_len] = curlastz[idx] + np.cumsum(tmp_e - tmp_f)

    return curlastz


def calc_curdistance(curlastz, meanx, sigmax, idx, profile_len, m, 
                     curdistance):
    tmp_a = curlastz[idx:profile_len+1]
    tmp_b = meanx[idx:profile_len]
    tmp_c = meanx[0:profile_len-idx]
    tmp_d = sigmax[idx:profile_len]
    tmp_e = sigmax[0:profile_len-idx]
    tmp_f = tmp_b * tmp_c
    tmp_g = (m-(tmp_a - m * tmp_f) / (tmp_d * tmp_e))
    curdistance[idx:profile_len] = np.sqrt(np.abs(2 * tmp_g))

    return curdistance


def time_is_exceeded(start_time, runtime):
    """Helper method to determine if the runtime has exceeded or not.

    Returns
    -------
    bool
        Whether or not hte runtime has exceeded.
    """
    elapsed = time.time() - start_time
    exceeded = runtime is not None and elapsed >= runtime
    if exceeded:
        warnings.warn(
            'Max runtime exceeded. Approximate solution is given.',
            RuntimeWarning
        )

    return exceeded


def scrimp_plus_plus(ts, m, step_size=0.25, runtime=None, random_state=None):
    """SCRIMP++ is an anytime algorithm that computes the matrix profile for a 
    given time series (ts) over a given window size (m). Essentially, it allows
    for an approximate solution to be provided for quicker analysis. In the 
    case of this implementation, the runtime is measured based on the wall 
    clock. If the number of seconds exceeds the runtime, then the approximate
    solution is returned. If the runtime is None, the exact solution is 
    returned.

    This algorithm was created at the University of California Riverside. For
    further academic understanding, please review this paper:

    Matrix Proﬁle XI: SCRIMP++: Time Series Motif Discovery at Interactive
    Speed. Yan Zhu, Chin-Chia Michael Yeh, Zachary Zimmerman, Kaveh Kamgar
    Eamonn Keogh, ICDM 2018.

    https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf

        Parameters
        ----------
        ts : np.ndarray
            The time series to compute the matrix profile for.
        m : int
            The window size.
        step_size : float, default 0.25
            The sampling interval for the window. The paper suggest 0.25 is the
            most practical. It should be a float value between 0 and 1.
        runtime : int, default None
            The maximum number of seconds based on wall clock time for this
            algorithm to run. It computes the exact solution when it is set to
            None.
        random_state : int, default None
            Set the random seed generator for reproducible results.

        Returns
        -------
        (np.array, np.array)
            The matrix profile and the matrix profile index respectively.
    """
    # start the timer here
    start_time = time.time()

    # validate step_size
    if not isinstance(step_size, float) or step_size > 1 or step_size < 0:
        raise ValueError('step_size should be a float between 0 and 1.')

    # validate runtime
    if runtime is not None and (not isinstance(runtime, int) or runtime < 1):
        raise ValueError('runtime should be a valid positive integer.')

    # validate random_state
    if random_state is not None:
        try:
            np.random.seed(random_state)
        except:
            raise ValueError('Invalid random_state value given.')
    
    ts_len = len(ts)

    # set the trivial match range
    exclusion_zone = calc_exclusion_zone(m)

    # value checking
    if m > ts_len / 2:
        raise ValueError('Time series is too short relative to desired \
            subsequence length')

    if m < 4:
        raise ValueError('Window size must be at least 4')

    # initialization
    step_size = calc_step_size(m, step_size)
    profile_len = calc_profile_len(ts_len, m)

    matrix_profile = np.zeros(profile_len)
    mp_index = np.zeros(profile_len, dtype='int32')

    X, n, sumx2, sumx, meanx, sigmax2, sigmax = fast_find_nn_pre(ts, m)

    ###########################
    # PreSCRIMP
    #
    # compute distance profile
    dotproduct = np.zeros(profile_len)
    refine_distance = np.full(profile_len, np.inf)
    orig_index = np.arange(profile_len)

    compute_order = list(range(0, profile_len, step_size))
    np.random.shuffle(compute_order)

    for iteration, idx in enumerate(compute_order):
        # compute distance profile
        subsequence = next_subsequence(ts, idx, m)
        
        distance_profile = calc_distance_profile(X, subsequence, n, m, meanx,
                                                 sigmax)
        
        # apply exclusion zone
        distance_profile = apply_exclusion_zone(
            idx, exclusion_zone, profile_len, distance_profile)
        
        # find and store nearest neighbor
        matrix_profile, mp_index, idx_nn = find_and_store_nn(
            iteration, idx, matrix_profile, mp_index, distance_profile)

        idx_diff = calc_idx_diff(idx, idx_nn)
        dotproduct = calc_dotproduct_idx(dotproduct, m, matrix_profile, idx,
                                         sigmax, idx_nn, meanx)

        endidx = calc_end_idx(profile_len, idx, step_size, idx_diff)

        dotproduct = calc_dotproduct_end_idx(ts, dotproduct, idx, m,
                                             endidx, idx_nn, idx_diff)

        refine_distance = calc_refine_distance_end_idx(
            refine_distance, dotproduct, idx, endidx, meanx, sigmax, idx_nn,
            idx_diff, m)
        
        beginidx = calc_begin_idx(idx, step_size, idx_diff)

        dotproduct = calc_dotproduct_begin_idx(
            ts, dotproduct, beginidx, idx, idx_diff, m, idx_nn)

        refine_distance = calc_refine_distance_begin_idx(
            refine_distance, dotproduct, beginidx, idx, idx_diff, idx_nn, 
            sigmax, meanx, m)

        matrix_profile, mp_index = apply_update_positions(matrix_profile, 
                                                          mp_index, 
                                                          refine_distance, 
                                                          beginidx, 
                                                          endidx, 
                                                          orig_index, idx_diff)

        # check if time is up
        if time_is_exceeded(start_time, runtime):            
            break

    if not time_is_exceeded(start_time, runtime):
        ###########################
        # SCRIMP
        #
        compute_order = orig_index[orig_index > exclusion_zone]
        np.random.shuffle(compute_order)

        curlastz = np.zeros(profile_len)
        curdistance = np.zeros(profile_len)
        dist1 = np.full(profile_len, np.inf)
        dist2 = np.full(profile_len, np.inf)

        for idx in compute_order:
            curlastz = calc_curlastz(ts, m, n, idx, profile_len, curlastz)
            curdistance = calc_curdistance(curlastz, meanx, sigmax, idx, 
                                           profile_len, m, curdistance)

            dist1[0:idx-1] = np.inf
            dist1[idx:profile_len] = curdistance[idx:profile_len]

            dist2[0:profile_len - idx] = curdistance[idx:profile_len]
            dist2[profile_len - idx + 2:profile_len] = np.inf

            loc1 = dist1 < matrix_profile
            if loc1.any():
                matrix_profile[loc1] = dist1[loc1]
                mp_index[loc1] = orig_index[loc1] - idx + 1

            loc2 = dist2 < matrix_profile
            if loc2.any():
                matrix_profile[loc2] = dist2[loc2]
                mp_index[loc2] = orig_index[loc2] + idx - 1

            # check if time is up
            if time_is_exceeded(start_time, runtime):             
                break

    return (matrix_profile, mp_index)