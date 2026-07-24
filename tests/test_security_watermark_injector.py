# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWatermarkInjector from former test_security.py

"""Focused suite: TestWatermarkInjector from former test_security.py."""

from __future__ import annotations

from tests.security_support import *  # noqa: F403


class TestWatermarkInjector:
    """Test suite for model watermarking/fingerprinting."""

    def test_inject_backdoor_success(self):
        """Successfully inject watermark into layer weights."""
        layer = MockLayer(n_neurons=10, n_inputs=5)
        trigger = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        target_idx = 3

        WatermarkInjector.inject_backdoor(layer, trigger, target_idx)

        assert np.array_equal(layer.weights[target_idx], trigger)
        assert layer._refresh_called is True

    def test_inject_backdoor_no_weights_raises(self):
        """Injection should fail if layer has no weights."""
        layer = MockLayerNoWeights()
        trigger = np.array([1.0, 0.0, 1.0])

        with pytest.raises(ValueError, match="no weights"):
            WatermarkInjector.inject_backdoor(layer, trigger, 0)

    def test_inject_backdoor_shape_mismatch_raises(self):
        """Injection should fail if trigger shape doesn't match inputs."""
        layer = MockLayer(n_neurons=10, n_inputs=5)
        trigger = np.array([1.0, 0.0, 1.0])  # Wrong shape (3 != 5)

        with pytest.raises(ValueError, match="shape mismatch"):
            WatermarkInjector.inject_backdoor(layer, trigger, 0)

    def test_verify_watermark_high_activation(self):
        """Watermarked neuron should have high activation for trigger."""
        layer = MockLayer(n_neurons=10, n_inputs=5)
        trigger = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        target_idx = 3

        WatermarkInjector.inject_backdoor(layer, trigger, target_idx)
        activation = WatermarkInjector.verify_watermark(layer, trigger, target_idx)

        # Perfect alignment should give 0.6 (mean of [1,0,1,0,1])
        assert activation == pytest.approx(0.6, abs=0.01)

    def test_verify_watermark_low_activation_no_watermark(self):
        """Non-watermarked neuron should have lower activation for trigger."""
        layer = MockLayer(n_neurons=10, n_inputs=5)
        trigger = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Don't inject, check random weights
        other_idx = 5
        activation = WatermarkInjector.verify_watermark(layer, trigger, other_idx)

        # Random weights should give ~0.5 activation on average
        assert 0.0 <= activation <= 1.0

    def test_watermark_preserves_other_neurons(self):
        """Watermarking one neuron should not affect others."""
        layer = MockLayer(n_neurons=10, n_inputs=5)
        original_weights = layer.weights.copy()
        trigger = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        target_idx = 3

        WatermarkInjector.inject_backdoor(layer, trigger, target_idx)

        # Check other neurons unchanged
        for i in range(10):
            if i != target_idx:
                assert np.array_equal(layer.weights[i], original_weights[i])
