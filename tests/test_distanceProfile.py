from matrixprofile.distanceProfile import *
import numpy as np
import pytest

class TestClass(object):

    def test_naiveDistanceProfile_self(self):
        outcome = (np.array([0.0,2.828,np.inf,np.inf,np.inf,np.inf,np.inf,2.828,0.0]),np.array([4.,4.,4.,4.,4.,4.,4.,4.,4.]))

        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])

        assert(np.round(naiveDistanceProfile(b,4,4),3) == outcome).all()

        #Need to confirm that we're not updating the original variable via shared memory
        assert(b == np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])).all()


    def test_naiveDistanceProfile_tsa_tsb(self):
        outcome = (np.array([0.0,2.828,4.0,2.828,0.0,2.828,4.0,2.828,0.0]),np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.]))

        a = np.array([0.0,1.0,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])

        assert(np.round(naiveDistanceProfile(a,0,4,b),3) == outcome).all()
        assert(a == np.array([0.0,1.0,1.0,0.0])).all()



    def test_massDistanceProfile_self(self):
        outcome = (np.array([0.0,2.828,np.inf,np.inf,np.inf,np.inf,np.inf,2.828,0.0]),np.array([4.,4.,4.,4.,4.,4.,4.,4.,4.]))

        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])

        assert(np.round(massDistanceProfile(b,4,4),3) == outcome).all()

        #Need to confirm that we're not updating the original variable via shared memory
        assert(b == np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])).all()


    def test_massDistanceProfile_tsa_tsb(self):
        outcome = (np.array([0.0,2.828,4.0,2.828,0.0,2.828,4.0,2.828,0.0]),np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.]))

        a = np.array([0.0,1.0,1.0,0.0])
        b = np.array([0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0])

        assert(np.round(massDistanceProfile(a,0,4,b),3) == outcome).all()
        assert(a == np.array([0.0,1.0,1.0,0.0])).all()
