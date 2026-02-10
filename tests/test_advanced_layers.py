
import unittest
import numpy as np
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer
from sc_neurocore.layers.sc_learning_layer import SCLearningLayer
from sc_neurocore.layers.sc_conv_layer import SCConv2DLayer
from sc_neurocore.layers.recurrent import SCRecurrentLayer

class TestAdvancedLayers(unittest.TestCase):
    
    def test_vectorized_layer(self):
        # 10 inputs -> 5 neurons
        layer = VectorizedSCLayer(n_inputs=10, n_neurons=5, length=64)
        inputs = np.random.random(10)
        output = layer.forward(inputs)
        self.assertEqual(output.shape, (5,))
        self.assertTrue(np.all(output >= 0.0))
        # Output is sum of (inputs * weights), max is n_inputs
        self.assertTrue(np.all(output <= 10.0))

    def test_learning_layer(self):
        layer = SCLearningLayer(n_inputs=5, n_neurons=2, length=64)
        inputs = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        spikes = layer.run_epoch(inputs)
        self.assertEqual(spikes.shape, (2, 64))
        
    def test_conv_layer(self):
        # 1 channel input, 2 filters, 3x3 kernel
        # Image 5x5
        layer = SCConv2DLayer(in_channels=1, out_channels=2, kernel_size=3, padding=0, length=64)
        img = np.random.random((1, 5, 5))
        out = layer.forward(img)
        # Output dim = 5-3+1 = 3x3
        self.assertEqual(out.shape, (2, 3, 3))

    def test_recurrent_layer(self):
        layer = SCRecurrentLayer(n_inputs=4, n_neurons=3, length=64)
        inp = np.array([0.5, 0.5, 0.5, 0.5])
        # Run 2 steps
        s1 = layer.step(inp)
        s2 = layer.step(inp)
        self.assertEqual(s1.shape, (3,))
        self.assertEqual(s2.shape, (3,))
        self.assertNotEqual(np.sum(s1), np.sum(s2)) # Should evolve

if __name__ == '__main__':
    unittest.main()
