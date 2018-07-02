# -*- coding: utf-8 -*-


"""
Loss functions.
"""

import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty


class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def errorString(self):
        pass

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass

    @abstractmethod
    def calculateDerivative(self, target, output):
        # calculate the error between target and output
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """

    def errorString(self):
        self.errorString = 'absolute'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return abs(target - output)

    def calculateDerivative(self, target, output):
        pass


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """

    def errorString(self):
        self.errorString = 'different'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return target - output

    def calculateDerivative(self, target, output):
        return -1


class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """

    def errorString(self):
        self.errorString = 'mse'

    def calculateError(self, target, output):
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        n = np.asarray(target).size
        return (1.0 / n) * np.sum((target - output) ** 2)

    def calculateDerivative(self, target, output):
        # MSEPrime = -n/2*(target - output)
        n = np.asarray(target).size
        return (2.0 / n) * (output - target)


class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """

    def errorString(self):
        self.errorString = 'sse'

    def calculateError(self, target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        return 0.5 * np.sum((target - output) ** 2)

    def calculateDerivative(self, target, output):
        # SSEPrime = -(target - output)
        return output - target


class BinaryCrossEntropyError(Error):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """

    def errorString(self):
        self.errorString = 'bce'

    def calculateError(self, target, output):
        return np.sum(target * np.log(output) + (1 - target) * np.log(1 - output))

    def calculateDerivative(self, target, output):
        # BCEPrime = -target/output + (1-target)/(1-output)
        return -target / output + (1 - target) / (1 - output)


class CrossEntropyError(Error):
    """
    The Loss calculated by the more general Cross Entropy between two
    probabilistic distributions.
    """

    def errorString(self):
        self.errorString = 'crossentropy'

    def calculateError(self, target, output):
        return -np.sum(target * np.log(output))

    def calculateDerivative(self, target, output):
        return output - target
