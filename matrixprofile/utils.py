import numpy as np
import numpy.fft as fft

def zNormalize(ts):
    """Return a z-normalized version of the time series"""

    ts -= np.mean(ts)
    std = np.std(ts)

    if std == 0:
        raise ValueError("The Standard Deviation cannot be zero")
    else:
        ts /= std

    return ts

def zNormalizeEuclidian(tsA,tsB):
    """Return the z-normalized Euclidian distance between the time series tsA and tsB"""

    if len(tsA) != len(tsB):
        raise ValueError("tsA and tsB must be the same length")

    return np.linalg.norm(zNormalize(tsA.astype("float64")) - zNormalize(tsB.astype("float64")))

def movstd(ts,m):
    """Calculate the standard deviation within a moving window of width m passing across the time series ts"""
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    #Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts),0,0)
    #Add zero to the beginning of the cumsum of ts ** 2
    sSq = np.insert(np.cumsum(ts ** 2),0,0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] -sSq[:-m]

    return np.sqrt(segSumSq / m - (segSum/m) ** 2)

def slidingDotProduct(query,ts):
    """Calculate the dot product between the query and all subsequences of length(query) in the timeseries ts. Note that we use Numpy's rfft method instead of fft."""

    m = len(query)
    n = len(ts)


    #If length is odd, zero-pad time time series
    ts_add = 0
    if n%2 ==1:
        ts = np.insert(ts,0,0)
        ts_add = 1

    q_add = 0
    #If length is odd, zero-pad query
    if m%2 == 1:
        query = np.insert(query,0,0)
        q_add = 1

    #This reverses the array
    query = query[::-1]


    query = np.pad(query,(0,n-m+ts_add-q_add),'constant')

    #Determine trim length for dot product. Note that zero-padding of the query has no effect on array length, which is solely determined by the longest vector
    trim = m-1+ts_add

    dot_product = fft.irfft(fft.rfft(ts)*fft.rfft(query))


    #Note that we only care about the dot product results from index m-1 onwards, as the first few values aren't true dot products (due to the way the FFT works for dot products)
    return dot_product[trim :]

def mass(query,ts):
    """Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS) between a query and timeseries. MASS is a Euclidian distance similarity search algorithm. Note that Z-normalization of the query changes mu(query) to 0 and sigma(query) to 1, which greatly simplifies the MASS formula described in Yeh et.al"""

    query_normalized = zNormalize(np.copy(query))
    m = len(query)
    std = movstd(ts,m)
    dot = slidingDotProduct(query_normalized,ts)


    return np.sqrt(2*(m-(dot/std)))
