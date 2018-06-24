import unittest

from mlp import MultilayerPerceptron


class MlpTests(unittest.TestCase):
    def test_compute_error(self):
        multilayer_perceptron = MultilayerPerceptron()
        self.assertEqual(1, 2)

    def test_update_weights(self):
        multilayer_perceptron = MultilayerPerceptron()
        self.assertEqual(1, 2)


if __name__ == '__main__':
    unittest.main()
