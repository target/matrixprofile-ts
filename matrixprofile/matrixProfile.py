from . import distanceProfile
from . import order
from .utils import mass
import numpy as np

def _matrixProfile(tsA,m,orderClass,distanceProfileFunction,tsB=None):
    order = orderClass(len(tsA)-m+1)

    #Account for the case where tsB is None (note that tsB = None triggers a self matrix profile)
    if tsB is None:
        mp = np.full(len(tsA)-m+1,np.inf)
        mpIndex = np.full(len(tsA)-m+1,np.inf)

    else:
        mp = np.full(len(tsB)-m+1,np.inf)
        mpIndex = np.full(len(tsB)-m+1,np.inf)

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


def _matrixProfile_sampling(tsA,m,orderClass,distanceProfileFunction,tsB=None,sampling=0.2):
    order = orderClass(len(tsA)-m+1)

    #Account for the case where tsB is None (note that tsB = None triggers a self matrix profile)
    if tsB is None:
        mp = np.full(len(tsA)-m+1,np.inf)
        mpIndex = np.full(len(tsA)-m+1,np.inf)

    else:
        mp = np.full(len(tsB)-m+1,np.inf)
        mpIndex = np.full(len(tsB)-m+1,np.inf)

    idx=order.next()

    #Define max numbers of iterations to sample
    iters = (len(tsA)-m+1)*sampling

    iter = 0

    while iter <= iters:
        (distanceProfile,querySegmentsID) = distanceProfileFunction(tsA,idx,m,tsB)

        #Check which of the indices have found a new minimum
        idsToUpdate = distanceProfile < mp

        #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mpIndex[idsToUpdate] = querySegmentsID[idsToUpdate]

        #Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp,distanceProfile)
        idx = order.next()

        iter += 1
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
    return _matrixProfile(tsA,m,order.linearOrder,distanceProfile.naiveDistanceProfile,tsB)

def stmp(tsA,m,tsB=None):
    return _matrixProfile(tsA,m,order.linearOrder,distanceProfile.massDistanceProfile,tsB)

def stamp(tsA,m,tsB=None,sampling=0.2):
    return _matrixProfile_sampling(tsA,m,order.randomOrder,distanceProfile.massDistanceProfile,tsB,sampling=sampling)



if __name__ == "__main__":
    import doctest
    doctest.method()
