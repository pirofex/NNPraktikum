# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp, transpose, matrix, add
from numpy import divide
from numpy import ones
from numpy import asarray
from numpy import dot
from numpy import diag


class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        # use e^x from numpy to avoid overflow
        return 1/(1+exp(-1.0*netOutput))

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)

        # tried to used vector multiplication, should work
        return dot(transpose(matrix(netOutput)), add(1,matrix((-1)*netOutput)))

    @staticmethod
    def tanh(netOutput):
        # return 2*Activation.sigmoid(2*netOutput)-1
        ex = exp(1.0*netOutput)
        exn = exp(-1.0*netOutput)
        return divide(ex-exn, ex+exn)  # element-wise division

    @staticmethod
    def tanhPrime(netOutput):
        # Here you have to code the derivative of tanh function
        return (1-Activation.tanh(netOutput)**2)

    @staticmethod
    def rectified(netOutput):
        return asarray([max(0.0, i) for i in netOutput])

    @staticmethod
    def rectifiedPrime(netOutput):
        # reluPrime=1 if netOutput > 0 otherwise 0
        #print(type(netOutput))
        return netOutput>0

    @staticmethod
    def identity(netOutput):
        return netOutput

    @staticmethod
    def identityPrime(netOutput):
        # identityPrime = 1
        return ones(netOutput.size)

    @staticmethod
    def softmax(netOutput):
        # normalize with max(netOutput), because exp() is not stable
        netOutput_exp = [exp(i - max(netOutput)) for i in netOutput]
        sum_netOutput_exp = sum(netOutput_exp)
        return [i / sum_netOutput_exp for i in netOutput_exp]
        
    @staticmethod
    def softmaxPrime(netOutput):
        # Implementation after explanation on https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative
        # turns all the vectors entries to e^(entry + max)
        outputPrime = map(lambda x: exp(x + max(netOutput)), netOutput)
        # divides each entry by the sum of all entries
        outputPrime = dot(outputPrime, 1/sum(outputPrime))
        # multiplies the vectors so we get the correct entries except for the diagonal
        jacobian = dot(transpose(outputPrime), dot(outputPrime, -1))
        # fix the diagonal so we get the correct derivative
        jacobian += diag(outputPrime)
        return jacobian

        """jacobian_matrix = diag(netOutput)

        for i in range(len(jacobian_matrix)):
            for j in range(len(jacobian_matrix)):
                if i == j:
                    jacobian_matrix[i][i] = netOutput[i] * (1 - netOutput[i])
                else:
                    jacobian_matrix[i][j] = netOutput[i] * (-netOutput[j])
        return jacobian_matrix """""
        
    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'softmax':
            return Activation.softmaxPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
