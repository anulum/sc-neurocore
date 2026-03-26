# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for quantization-aware training module

from __future__ import annotations

import numpy as np

from sc_neurocore.qat import QuantizedSNNLayer, quantize_aware_train_step, TernaryWeights
from sc_neurocore.qat.quantize import _ste_quantize


class TestSteQuantize:
    def test_symmetric_roundtrip(self):
        x = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        q = _ste_quantize(x, bits=8, symmetric=True)
        assert q.shape == x.shape
        np.testing.assert_allclose(q, x, atol=0.01)

    def test_asymmetric(self):
        x = np.array([0.1, 0.5, 0.9, 1.3])
        q = _ste_quantize(x, bits=4, symmetric=False)
        assert q.shape == x.shape
        assert q.min() >= x.min() - 1e-6
        assert q.max() <= x.max() + 1e-6

    def test_quantize_reduces_unique_values(self):
        x = np.random.randn(100)
        q = _ste_quantize(x, bits=4, symmetric=True)
        assert len(np.unique(q)) <= 2**4

    def test_zero_preserved(self):
        x = np.array([0.0])
        assert _ste_quantize(x, bits=8)[0] == 0.0


class TestQuantizedSNNLayer:
    def test_forward_shape(self):
        layer = QuantizedSNNLayer(n_inputs=4, n_neurons=3, weight_bits=8)
        out = layer.forward(np.random.rand(4))
        assert out.shape == (3,)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_export_weights_quantized(self):
        layer = QuantizedSNNLayer(n_inputs=4, n_neurons=3, weight_bits=4)
        q = layer.export_weights()
        assert q.shape == (3, 4)
        assert len(np.unique(q)) <= 2**4

    def test_train_step(self):
        layer = QuantizedSNNLayer(n_inputs=4, n_neurons=2, weight_bits=8)
        w_before = layer.W.copy()
        result = quantize_aware_train_step(
            layer, np.random.rand(4), np.array([1.0, 0.0])
        )
        assert result["loss"] >= 0
        assert not np.array_equal(layer.W, w_before)

    def test_multiple_steps_reduce_loss(self):
        layer = QuantizedSNNLayer(n_inputs=4, n_neurons=2, weight_bits=8)
        x = np.array([1.0, 0.0, 1.0, 0.0])
        target = np.array([1.0, 0.0])
        losses = []
        for _ in range(20):
            result = quantize_aware_train_step(layer, x, target, lr=0.05)
            losses.append(result["loss"])
        assert losses[-1] <= losses[0]

    def test_deterministic(self):
        a = QuantizedSNNLayer(n_inputs=4, n_neurons=3, weight_bits=8)
        b = QuantizedSNNLayer(n_inputs=4, n_neurons=3, weight_bits=8)
        np.testing.assert_array_equal(a.W, b.W)


class TestTernaryWeights:
    def test_ternary_values(self):
        tw = TernaryWeights()
        t = tw.quantize(np.random.randn(10, 10))
        assert set(np.unique(t)).issubset({-1.0, 0.0, 1.0})

    def test_sparsity(self):
        tw = TernaryWeights(threshold_ratio=0.5)
        s = tw.sparsity(np.random.randn(100, 100))
        assert 0 < s < 1

    def test_higher_threshold_more_sparse(self):
        w = np.random.randn(100, 100)
        low = TernaryWeights(threshold_ratio=0.3).sparsity(w)
        high = TernaryWeights(threshold_ratio=0.9).sparsity(w)
        assert high > low

    def test_all_zero_input(self):
        tw = TernaryWeights()
        t = tw.quantize(np.zeros((5, 5)))
        np.testing.assert_array_equal(t, 0.0)
