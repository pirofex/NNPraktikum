#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=True)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)
    # Original
    #myLRClassifier = LogisticRegression(data.trainingSet,
    #                                    data.validationSet,
    #                                    data.testSet,
    #                                    learningRate=0.005,
    #                                    epochs=30)

    # Modified (fun)
    myLRClassifier = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.003,
                                        epochs=80,
                                        loss='crossentropy')
                                        
    
    # Report the result #
    print("=========================")
    evaluator = Evaluator()                                        

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nPerceptron has been training..")
    myPerceptronClassifier.train()
    print("Done..")
    
    print("\nLogistic Regression has been training..")
    myLRClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    perceptronPred = myPerceptronClassifier.evaluate()
    lrPred = myLRClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    #evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    #evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, perceptronPred)
    
    print("\nResult of the Logistic Regression recognizer:")
    #evaluator.printComparison(data.testSet, lrPred)    
    evaluator.printAccuracy(data.testSet, lrPred)
    
    # Draw
    plot = PerformancePlot("Logistic Regression validation")
    plot.draw_performance_epoch(myLRClassifier.performances,
                                myLRClassifier.epochs)
    
    
if __name__ == '__main__':
    main()
