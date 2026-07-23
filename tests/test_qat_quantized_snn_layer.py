# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantizedSNNLayer from former test_qat.py

"""Focused suite: TestQuantizedSNNLayer from former test_qat.py."""

from __future__ import annotations

from tests.qat_support import *  # noqa: F403

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
        result = quantize_aware_train_step(layer, np.random.rand(4), np.array([1.0, 0.0]))
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
