#!/usr/bin/env python
# -*- coding: utf-8 -*-
from activation_functions import Activation
import numpy as np
import unittest


class SoftmaxTest(unittest.TestCase):

    def test_softmax1(self):
        z = [1, 2]
        result = np.round(Activation.softmax(np.array(z)), 3)
        expected_result = np.array([0.269,  0.731])
        self.assertItemsEqual(expected_result, result)

    def test_softmax2(self):
        # example from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative
        z = [1, 2, 3]
        result = np.round(Activation.softmax(np.array(z)), 3)
        expected_result = np.array([0.090, 0.245, 0.665])
        self.assertItemsEqual(expected_result, result)

    def test_softmax3(self):
        # example from https://en.wikipedia.org/wiki/Softmax_function
        z = [1, 2, 3, 4, 1, 2, 3]
        result = np.round(Activation.softmax(np.array(z)), 3)
        expected_result = np.array([0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175])
        self.assertItemsEqual(expected_result, result)

    def test_softmax4(self):
        # example from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative
        z = [1000, 2000, 3000]
        result = np.round(Activation.softmax(np.array(z)), 3)
        expected_result = np.array([0., 0., 1.])
        self.assertItemsEqual(expected_result, result)

    def test_softmax_prime1(self):
        z = [1, 2]
        result = Activation.softmaxPrime(Activation.softmax(np.array(z)))
        expected_result = np.array([
            [ 0.197, -0.197],
            [-0.197,  0.197]
        ])
        self.assertItemsEqual(expected_result.flatten(), result.flatten().round(3))

    def test_softmax_prime2(self):
        z = [1, 2, 3]
        result = Activation.softmaxPrime(Activation.softmax(np.array(z)))
        expected_result = np.array([
            [ 0.082, -0.022, - 0.060],
            [-0.022,  0.185, - 0.163],
            [-0.060, -0.163,   0.223]
        ])
        self.assertItemsEqual(expected_result.flatten(), result.flatten().round(3))

    def test_softmax_prime3(self):
        z = [1, 2, 3, 4, 1, 2, 3]
        result = Activation.softmaxPrime(Activation.softmax(np.array(z)))
        expected_result = np.array([
            [ 0.023, -0.002, -0.004, -0.011, -0.001, -0.002, -0.004],
            [-0.002,  0.060, -0.011, -0.031, -0.002, -0.004, -0.011],
            [-0.004, -0.011,  0.144, -0.083, -0.004, -0.011, -0.031],
            [-0.011, -0.031, -0.083,  0.249, -0.011, -0.031, -0.083],
            [-0.001, -0.002, -0.004, -0.011,  0.023, -0.002, -0.004],
            [-0.002, -0.004, -0.011, -0.031, -0.002,  0.060, -0.011],
            [-0.004, -0.011, -0.031, -0.083, -0.004, -0.011,  0.144]
        ])
        self.assertItemsEqual(expected_result.flatten(), result.flatten().round(3))

    def test_softmax_prime4(self):
        z = [1000, 2000, 3000]
        result = Activation.softmaxPrime(Activation.softmax(np.array(z)))
        expected_result = np.array([
            [ 0., -0., -0.],
            [-0.,  0., -0.],
            [-0., -0.,  0.]
        ])
        self.assertItemsEqual(expected_result.flatten(), result.flatten().round(3))


if __name__ == '__main__':
    unittest.main()
