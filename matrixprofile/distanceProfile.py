from .utils import *
import numpy as np

def naiveDistanceProfile(tsA,idx,m,tsB = None):
    '''Return the distance profile of a query within tsA against the time series tsB. Uses the naive all-pairs comparison. idx defines the starting index of the query within tsA and m is the length of the query.'''

    selfJoin = False
    if tsB is None:
        selfJoin = True
        tsB = tsA

    print('it works')

    query = tsA[idx: (idx+m)]
    distanceProfile = []
    n = len(tsB)

    for i in range(n-m+1):
        distanceProfile.append(zNormalizeEuclidian(query,tsB[i:i+m]))

    dp = np.array(distanceProfile)

    if selfJoin:
        trivialMatchRange = (int(max(0,idx - np.round(m/2,0))),int(min(idx + np.round(m/2+1,0),n)))

        dp[trivialMatchRange[0]: trivialMatchRange[1]] = np.inf

    return (dp,np.full(n-m+1,idx,dtype=float))


def massDistanceProfile(tsA,idx,m,tsB = None):
    '''Return the distance profile of a query within tsA against the time series tsB. Uses the more efficient MASS comparison. idx defines the starting index of the query within tsA and m is the length of the query.'''


    selfJoin = False
    if tsB is None:
        selfJoin = True
        tsB = tsA

    query = tsA[idx:(idx+m)]
    n = len(tsB)
    distanceProfile = mass(query,tsB)
    if selfJoin:
        trivialMatchRange = (int(max(0,idx - np.round(m/2,0))),int(min(idx + np.round(m/2+1,0),n)))
        distanceProfile[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

    #Both the distance profile and corresponding matrix profile index (which should just have the current index)
    return (distanceProfile,np.full(n-m+1,idx,dtype=float))

if __name__ == "__main__":
    import doctest
    doctest.method()
