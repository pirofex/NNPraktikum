import numpy as np
import logging
import util.loss_functions as loss_functions

from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        # self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        if loss == 'bce':
            self.loss = loss_functions.BinaryCrossEntropyError()
        elif loss == 'crossentropy':
            self.loss = loss_functions.CrossEntropyError()
        elif loss == 'sse':
            self.loss = loss_functions.SumSquaredError()
        elif loss == 'mse':
            self.loss = loss_functions.MeanSquaredError()
        elif loss == 'different':
            self.loss = loss_functions.DifferentError()
        elif loss == 'absolute':
            self.loss = loss_functions.AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128,
                                         None, inputActivation, False))

        # self.layers.append(LogisticLayer(128, 128,
        #                                  None, "sigmoid", False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10,
                                         None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                           axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                             axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def valueToVector(self, vector, value, function):
        """
        Used to compare every entry of vector with the specified function to zero except for the entry at index
        value, which is compared to one. used in error calculation

        Parameters
        ----------
        vector : list-like
            a vector usually containing doubles to be compared
        value : int
            the index of the entry that is compared to one
        function : method
            a method calculating the error between the vector entries and one or zero

        # used to make error calculation easier
        """
        out = np.ndarray(np.shape(vector))
        for i in range(0, np.size(vector)):
            if i != value:
                out[i] = function(0, vector[i])
            else:
                out[i] = function(1, vector[i])
        return out

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        for layer in self.layers:
            new_inp = layer.forward(np.atleast_2d(inp))
            # inputs into layers require a leading 1 to represent the layer's bias
            inp = np.insert(new_inp, 0, 1, axis=1)

        # return the input without the bias for the output layer for the softmax to work
        return new_inp

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        # the target is only one for the correct label, otherwise zero
        outp = self._get_output_layer().outp
        target_vec = np.insert(np.zeros(outp.shape), target, 1, axis=1)
        return self.loss.calculateError(target_vec,outp)

    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """

        # backpropagation should probably be done here, meanwhile in train()

        for layer in self.layers:
            layer.updateWeights(learningRate)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))
            for data, label in zip(self.trainingSet.input,
                                   self.trainingSet.label):

                # Do a forward pass to calculate the output and the error
                outp = self._feed_forward(data)
                target_vec = np.insert(np.zeros((1, outp.size-1)), label, 1, axis=1)
                pred = np.argmax(outp, axis=1)
                # compute error in relation to the target input and calculate weight deltas. For more information see the
                # compute derivative function in logistic_layer
                next_delta = (self._get_output_layer()).computeDerivative(
                    (self.loss.calculateDerivative(target_vec,outp)), 1.0)
                next_layer = self._get_output_layer()
                # reverse iterate over the layers, skipping the output layer (layers(-1)) since it already has been
                # handled separately
                for i in range(2, (self.layers.__len__() + 1)):
                    # TODO i is always 2?! Better: for layer in reversed(self.layers)?
                    # back propagation for each layer using weights and derivatives of the previous one
                    next_delta = self._get_layer(-i).computeDerivative(next_delta, np.transpose(next_layer.weights[1:]))
                    next_layer = self._get_layer(-i)

                # Update weights in the online learning fashion
                self._update_weights(self.learningRate)

            if verbose:
                accuracy = accuracy_score(self.validationSet.label, self.evaluate(self.validationSet.input))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                # print(self.layers[0].weights)
                # print(self.layers[1].weights)
                print("-----------------------------")

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        out = self._feed_forward(test_instance)
        # return the index of the maximum, the index representing the number recognized
        return np.argmax(out, axis=1)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                             axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
