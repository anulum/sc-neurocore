
import unittest
import numpy as np
import os
from sc_neurocore.interfaces.bci import BCIDecoder
from sc_neurocore.export.onnx_exporter import SCOnnxExporter
from sc_neurocore.profiling.energy import EnergyMetrics
from sc_neurocore.security.watermark import WatermarkInjector
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

class TestSystems(unittest.TestCase):
    
    def test_bci_decoder(self):
        bci = BCIDecoder(channels=2)
        sig = np.array([[0.1], [0.9]])
        bits = bci.encode_to_bitstream(sig, length=10)
        self.assertEqual(bits.shape, (2, 10))
        
    def test_onnx_export(self):
        layer = VectorizedSCLayer(n_inputs=2, n_neurons=2)
        SCOnnxExporter.export([layer], "test_model.json")
        self.assertTrue(os.path.exists("test_model.json"))
        # Cleanup
        if os.path.exists("test_model.json"): os.remove("test_model.json")
        if os.path.exists("test_model.json_layer_0_weights.npy"): os.remove("test_model.json_layer_0_weights.npy")
        
    def test_energy_metrics(self):
        prof = EnergyMetrics()
        prof.total_ops_and = 1000
        e = prof.estimate_energy()
        self.assertGreater(e, 0.0)
        
    def test_watermarking(self):
        layer = VectorizedSCLayer(n_inputs=10, n_neurons=2)
        trigger = np.ones(10)
        WatermarkInjector.inject_backdoor(layer, trigger, 0)
        act = WatermarkInjector.verify_watermark(layer, trigger, 0)
        self.assertAlmostEqual(act, 1.0)

if __name__ == '__main__':
    unittest.main()
