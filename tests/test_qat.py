# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
from __future__ import annotations
import numpy as np
from sc_neurocore.qat import QuantizedSNNLayer, quantize_aware_train_step, TernaryWeights
from sc_neurocore.qat.quantize import _ste_quantize


class TestQuantizedSNNLayer:
    def test_forward(self):
        layer = QuantizedSNNLayer(n_inputs=4, n_neurons=3, weight_bits=8)
        out = layer.forward(np.random.rand(4))
        assert out.shape == (3,)

    def test_export(self):
        layer = QuantizedSNNLayer(n_inputs=4, n_neurons=3, weight_bits=4)
        q = layer.export_weights()
        assert q.shape == (3, 4)

    def test_train_step(self):
        layer = QuantizedSNNLayer(n_inputs=4, n_neurons=2, weight_bits=8)
        result = quantize_aware_train_step(layer, np.random.rand(4), np.array([1.0, 0.0]))
        assert result["loss"] >= 0


class TestTernaryWeights:
    def test_quantize(self):
        tw = TernaryWeights()
        t = tw.quantize(np.random.randn(10, 10))
        assert set(np.unique(t)).issubset({-1.0, 0.0, 1.0})

    def test_sparsity(self):
        tw = TernaryWeights(threshold_ratio=0.5)
        assert 0 < tw.sparsity(np.random.randn(100, 100)) < 1


class TestSteQuantize:
    def test_asymmetric(self):
        x = np.array([0.1, 0.5, 0.9, 1.3])
        q = _ste_quantize(x, bits=4, symmetric=False)
        assert q.shape == x.shape
        assert q.min() >= x.min() - 1e-6
        assert q.max() <= x.max() + 1e-6
