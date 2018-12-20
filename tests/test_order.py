from matrixprofile.order import *
import numpy as np
import pytest

class TestClass(object):

    def test_linearOrder_length(self):
        ord = linearOrder(10)

        t = 0
        indices = []

        while t is not None:
            indices.append(t)
            t = ord.next()

        assert(len(indices[1:]) == 10)


    def test_linearOrder_vals(self):
        ord = linearOrder(10)

        t = 0
        indices = []

        while t is not None:
            indices.append(t)
            t = ord.next()

        unique_vals = np.unique(indices[1:])
        outcome = np.array([0,1,2,3,4,5,6,7,8,9])

        assert(unique_vals == outcome).all()


    def test_randomOrder_length(self):
        ord = randomOrder(10)

        t = 0
        indices = []

        while t is not None:
            indices.append(t)
            t = ord.next()

        assert(len(indices[1:]) == 10)


    def test_randomOrder_vals(self):
        ord = randomOrder(10)

        t = 0
        indices = []

        while t is not None:
            indices.append(t)
            t = ord.next()

        unique_vals = np.unique(indices[1:])
        outcome = np.array([0,1,2,3,4,5,6,7,8,9])

        assert(unique_vals == outcome).all()
