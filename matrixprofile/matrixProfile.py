# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

from . import distanceProfile
from . import order
from .utils import mass, movmeanstd
import numpy as np
import multiprocessing
from functools import partial
import math

from .scrimp import scrimp_plus_plus

def _self_join_or_not_preprocess(tsA, tsB, m):
    """
    Core method for determining if a self join is occuring and returns appropriate
    profile and index numpy arrays with correct dimensions as all np.nan values.
    
    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    tsB: Time series to compare the query against. Note that, if no value is provided, ts_b = ts_a by default.
    m: Length of subsequence to compare.
    """
    n = len(tsA)
    if tsB is not None:
        n = len(tsB)
    
    shape = n - m + 1
    
    return (np.full(shape, np.inf), np.full(shape, np.inf))

def _matrixProfile(tsA,m,orderClass,distanceProfileFunction,tsB=None):
    """
    Core method for calculating the Matrix Profile

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    orderClass: Method defining the order in which distance profiles are calculated.
    distanceProfileFunction: Method for calculating individual distance profiles.
    sampling: The percentage of all possible distance profiles to sample for the final Matrix Profile.
    """

    order = orderClass(len(tsA)-m+1)
    mp, mpIndex = _self_join_or_not_preprocess(tsA, tsB, m)

    idx=order.next()
    while idx != None:
        (distanceProfile,querySegmentsID) = distanceProfileFunction(tsA,idx,m,tsB)

        #Check which of the indices have found a new minimum
        idsToUpdate = distanceProfile < mp

        #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mpIndex[idsToUpdate] = querySegmentsID[idsToUpdate]

        #Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp,distanceProfile)
        idx = order.next()

    return (mp,mpIndex)

def _stamp_parallel(tsA, m, tsB=None, sampling=0.2, n_threads=-1, random_state=None):
    """
    Computes distance profiles in parallel using all CPU cores by default.
    
    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    sampling: The percentage of all possible distance profiles to sample for the final Matrix Profile. 0 to 1
    n_threads: Number of threads to use in parallel mode. Defaults to using all CPU cores.
    random_state: Set the random seed generator for reproducible results.
    """
    if n_threads is -1:
        n_threads = multiprocessing.cpu_count()
    
    n = len(tsA)
    mp, mpIndex = _self_join_or_not_preprocess(tsA, tsB, m)

    # determine sampling size
    sample_size = math.ceil((n - m + 1) * sampling)
    
    # generate indices to sample and split based on n_threads
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(n - m + 1)
    indices = np.random.choice(indices, size=sample_size, replace=False)
    indices = np.array_split(indices, n_threads)
    
    # create pool of workers and compute
    with multiprocessing.Pool(processes=n_threads) as pool:
        func = partial(distanceProfile.mass_distance_profile_parallel, tsA=tsA, tsB=tsB, m=m)
        results = pool.map(func, indices)
    
    # The overall matrix profile is the element-wise minimum of each sub-profile, and each element of the overall
    # matrix profile index is the time series position of the corresponding sub-profile.
    for result in results:
        for dp, querySegmentsID in result:
            #Check which of the indices have found a new minimum
            idsToUpdate = dp < mp

            #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
            mpIndex[idsToUpdate] = querySegmentsID[idsToUpdate]

            #Update the matrix profile to include the new minimum values (where appropriate)
            mp = np.minimum(mp, dp)

    return (mp, mpIndex)

def _matrixProfile_sampling(tsA,m,orderClass,distanceProfileFunction,tsB=None,sampling=0.2,random_state=None):
    order = orderClass(len(tsA)-m+1, random_state=random_state)
    mp, mpIndex = _self_join_or_not_preprocess(tsA, tsB, m)

    idx=order.next()

    #Define max numbers of iterations to sample
    iters = (len(tsA)-m+1)*sampling

    iter_val = 0

    while iter_val < iters:
        (distanceProfile,querySegmentsID) = distanceProfileFunction(tsA,idx,m,tsB)

        #Check which of the indices have found a new minimum
        idsToUpdate = distanceProfile < mp

        #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mpIndex[idsToUpdate] = querySegmentsID[idsToUpdate]

        #Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp,distanceProfile)
        idx = order.next()

        iter_val += 1
    return (mp,mpIndex)


#Write matrix profile function for STOMP and then consolidate later! (aka link to the previous distance profile)
def _matrixProfile_stomp(tsA,m,orderClass,distanceProfileFunction,tsB=None):
    order = orderClass(len(tsA)-m+1)
    mp, mpIndex = _self_join_or_not_preprocess(tsA, tsB, m)

    idx=order.next()

    #Get moving mean and standard deviation
    mean, std = movmeanstd(tsA,m)

    #Initialize code to set dot_prev to None for the first pass
    dp = None

    #Initialize dot_first to None for the first pass
    dot_first = None

    while idx != None:

        #Need to pass in the previous sliding dot product for subsequent distance profile calculations
        (distanceProfile,querySegmentsID),dot_prev = distanceProfileFunction(tsA,idx,m,tsB,dot_first,dp,mean,std)

        if idx == 0:
            dot_first = dot_prev

        #Check which of the indices have found a new minimum
        idsToUpdate = distanceProfile < mp

        #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mpIndex[idsToUpdate] = querySegmentsID[idsToUpdate]

        #Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp,distanceProfile)
        idx = order.next()

        dp = dot_prev
    return (mp,mpIndex)

def stampi_update(tsA,m,mp,mpIndex,newval,tsB=None,distanceProfileFunction=distanceProfile.massDistanceProfile):
    '''Updates the self-matched matrix profile for a time series TsA with the arrival of a new data point newval. Note that comparison of two separate time-series with new data arriving will be built later -> currently, tsB should be set to tsA'''

    #Update time-series array with recent value
    tsA_new = np.append(np.copy(tsA),newval)

    #Expand matrix profile and matrix profile index to include space for latest point
    mp_new= np.append(np.copy(mp),np.inf)
    mpIndex_new = np.append(np.copy(mpIndex),np.inf)

    #Determine new index value
    idx = len(tsA_new)-m

    (distanceProfile,querySegmentsID) = distanceProfileFunction(tsA_new,idx,m,tsB)

    #Check which of the indices have found a new minimum
    idsToUpdate = distanceProfile < mp_new

    #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
    mpIndex_new[idsToUpdate] = querySegmentsID[idsToUpdate]

    #Update the matrix profile to include the new minimum values (where appropriate)
    mp_final = np.minimum(np.copy(mp_new),distanceProfile)

    #Finally, set the last value in the matrix profile to the minimum of the distance profile (with corresponding index)
    mp_final[-1] = np.min(distanceProfile)
    mpIndex_new[-1] = np.argmin(distanceProfile)

    return (mp_final,mpIndex_new)


def naiveMP(tsA,m,tsB=None):
    """
    Calculate the Matrix Profile using the naive all-pairs calculation.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    """
    return _matrixProfile(tsA,m,order.linearOrder,distanceProfile.naiveDistanceProfile,tsB)

def stmp(tsA,m,tsB=None):
    """
    Calculate the Matrix Profile using the more efficient MASS calculation. Distance profiles are computed linearly across every time series index.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    """
    return _matrixProfile(tsA,m,order.linearOrder,distanceProfile.massDistanceProfile,tsB)

def stamp(tsA,m,tsB=None,sampling=0.2, n_threads=None, random_state=None):
    """
    Calculate the Matrix Profile using the more efficient MASS calculation. Distance profiles are computed in a random order.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    sampling: The percentage of all possible distance profiles to sample for the final Matrix Profile. 0 to 1
    n_threads: Number of threads to use in parallel mode. Defaults to single threaded mode. Set to -1 to use all threads.
    random_state: Set the random seed generator for reproducible results.
    """
    if sampling > 1 or sampling < 0:
        raise ValueError('Sampling value must be a percentage in decimal format from 0 to 1.')
    
    if n_threads is None:
        return _matrixProfile_sampling(tsA,m,order.randomOrder,distanceProfile.massDistanceProfile,tsB,sampling=sampling,random_state=random_state)
    
    return _stamp_parallel(tsA, m, tsB=tsB, sampling=sampling, n_threads=n_threads, random_state=random_state)

def stomp(tsA,m,tsB=None):
    """
    Calculate the Matrix Profile using the more efficient MASS calculation. Distance profiles are computed according to the directed STOMP procedure.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    """
    return _matrixProfile_stomp(tsA,m,order.linearOrder,distanceProfile.STOMPDistanceProfile,tsB)



if __name__ == "__main__":
    import doctest
    doctest.method()
