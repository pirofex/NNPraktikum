import unittest

from mlp import MultilayerPerceptron


class MlpTests(unittest.TestCase):
    def test_compute_error(self):
        multilayer_perceptron = MultilayerPerceptron()
        self.assertEqual(1, 2)

    def test_update_weights(self):
        multilayer_perceptron = MultilayerPerceptron()
        self.assertEqual(1, 2)

    def test_feed_forward(self):
        train = []
        for i in range(127):
            train.append(1)

        valid = []
        test = []
        layers = None
        inputWeights = None
        outputTask = 'classification'
        outputActivation = 'softmax'
        loss = 'bce'
        learningRate = 0.01
        epochs = 50
        mlp = MultilayerPerceptron(train, valid, test, layers,
                                                      inputWeights, outputTask, outputActivation,
                                                      loss, learningRate, epochs)
        res = mlp._feed_forward(train)
        print(res)

        """only placeholder, i couldn't run the testcase"""
        exp_res = [1,2,3,4]
        self.assertItemsEqual(res, exp_res)



if __name__ == '__main__':
    unittest.main()
