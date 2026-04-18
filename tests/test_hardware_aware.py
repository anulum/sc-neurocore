# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for hardware-aware SC layer

"""Tests for HardwareAwareSCLayer with defect injection."""

import numpy as np

from sc_neurocore.layers.hardware_aware import HardwareAwareSCLayer


class TestHardwareAwareSCLayer:
    def test_forward_shape(self):
        layer = HardwareAwareSCLayer(n_inputs=4, n_neurons=3, length=64)
        out = layer.forward([0.3, 0.5, 0.7, 0.2])
        assert out.shape == (3,)

    def test_stuck_synapses_exist(self):
        layer = HardwareAwareSCLayer(n_inputs=10, n_neurons=10, stuck_rate=0.1)
        assert layer.n_stuck > 0
        assert layer.stuck_fraction > 0.0

    def test_stuck_weights_unchanged_after_update(self):
        layer = HardwareAwareSCLayer(n_inputs=4, n_neurons=3, stuck_rate=0.3, seed=42)
        stuck_before = layer.weights[layer.stuck_mask].copy()
        gradient = np.random.randn(3, 4)
        layer.update_weights(gradient, lr=0.1)
        stuck_after = layer.weights[layer.stuck_mask]
        np.testing.assert_array_equal(stuck_before, stuck_after)

    def test_non_stuck_weights_change(self):
        layer = HardwareAwareSCLayer(n_inputs=4, n_neurons=3, stuck_rate=0.0)
        before = layer.weights.copy()
        gradient = np.ones((3, 4)) * 0.5
        layer.update_weights(gradient, lr=0.1)
        assert not np.allclose(before, layer.weights)

    def test_weights_stay_in_bounds(self):
        layer = HardwareAwareSCLayer(n_inputs=4, n_neurons=3, stuck_rate=0.0)
        gradient = np.ones((3, 4)) * 100
        layer.update_weights(gradient, lr=1.0)
        assert np.all(layer.weights >= 0.0)
        assert np.all(layer.weights <= 1.0)

    def test_zero_stuck_rate(self):
        layer = HardwareAwareSCLayer(n_inputs=4, n_neurons=3, stuck_rate=0.0)
        assert layer.n_stuck == 0
