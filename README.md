[![PyPI version](https://badge.fury.io/py/matrixprofile-ts.svg)](https://badge.fury.io/py/matrixprofile-ts)
[![Build Status](https://travis-ci.org/target/matrixprofile-ts.png)](https://travis-ci.org/target/matrixprofile-ts)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# matrixprofile-ts

matrixprofile-ts is a Python 3 library for evaluating time series data using the matrix profile algorithms developed by the Keough and Mueen research groups at the University of California-Riverside and the University of New Mexico. Current algorithms implemented include MASS, STMP, STAMP, STAMPI and STOMP.

## Contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Detailed Example](#detailed-example)
- [Matrix Profile in Other Languages](#matrix-profile-in-other-languages)
- [Contact](#contact)
- [Citations](#citations)

## Installation

Major releases of matrixprofile-ts are available on the Python Package Index:

`pip install matrixprofile-ts`

Details about each release can be found [here](https://github.com/target/matrixprofile-ts/blob/master/docs/Releases.md).

## Quick start

```
>>> from matrixprofile import *
>>> import numpy as np
>>> a = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])
>>> matrixProfile.stomp(a,4)
(array([0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([4., 5., 6., 7., 0., 1., 2., 3., 0.]))
```
Note that STOMP is highly recommended for calculating the Matrix Profile, due to its speed.

## Detailed example

A Jupyter notebook containing code for this example can be found [here](https://github.com/target/matrixprofile-ts/blob/master/docs/Matrix_Profile_Tutorial.ipynb)

We can take a synthetic signal and use STOMP to calculate the corresponding Matrix Profile (this is the same synthetic signal as in the [Golang Matrix Profile library](https://github.com/aouyang1/go-matrixprofile))

![datamp](https://github.com/target/matrixprofile-ts/blob/master/datamp.png)


There are several items of note:

- The Matrix Profile value jumps at each phase change. High Matrix Profile values are associated with "discords": time series behavior that hasn't been observed before.

- Repeated patterns in the data (or "motifs") lead to low Matrix Profile values.


We can introduce an anomaly to the end of the time series and use STAMPI to detect it

![datampanom](https://github.com/target/matrixprofile-ts/blob/master/datampanom.png)

The Matrix Profile has spiked in value, highlighting the (potential) presence of a new behavior. Note that Matrix Profile anomaly detection capabilities will depend on the nature of the data, as well as the selected subquery length parameter. Like all good algorithms, it's important to try out different parameter values.



## Matrix Profile in Other Languages
- R: https://github.com/franzbischoff/tsmp
- Golang: https://github.com/aouyang1/go-matrixprofile

## Contact
- Andrew Van Benschoten (avbs89@gmail.com)

## Citations
1. Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen, Eamonn Keogh (2016). Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View that Includes Motifs, Discords and Shapelets. IEEE ICDM 2016

2. Matrix Profile II: Exploiting a Novel Algorithm and GPUs to break the one Hundred Million Barrier for Time Series Motifs and Joins.  Yan Zhu, Zachary Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh, Gareth Funning, Abdullah Mueen, Philip Berisk and Eamonn Keogh (2016). EEE ICDM 2016

3. Matrix Profile V: A Generic Technique to Incorporate Domain Knowledge into Motif Discovery. Hoang Anh Dau and Eamonn Keogh. KDD'17, Halifax, Canada.
