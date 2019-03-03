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

    def test_randomOrder_random_state_same_results(self):
        random_state=99
        order = randomOrder(10, random_state=random_state)
        order2 = randomOrder(10, random_state=random_state)
        
        indices = []
        t = order.next()
        while t is not None:
            indices.append(t)
            t = order.next()
            
        indices2 = []
        t2 = order2.next()
        while t2 is not None:
            indices2.append(t2)
            t2 = order2.next()
        
        indices = np.array(indices)
        indices2 = np.array(indices2)
        
        all_same = (indices == indices2).all()
        assert(all_same == True)