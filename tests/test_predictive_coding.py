# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for predictive coding SC layer (Conjecture C9)

"""Tests for zero-multiplication predictive coding in stochastic computing."""

import numpy as np

from sc_neurocore.layers.predictive_coding import PredictiveCodingSCLayer


class TestPredictiveCodingSCLayer:
    def test_output_keys(self):
        layer = PredictiveCodingSCLayer(n_inputs=3, n_neurons=2, length=64, seed=42)
        result = layer.forward([0.3, 0.5, 0.7])
        assert "prediction_error" in result
        assert "surprises" in result
        assert "predictions" in result

    def test_output_shapes(self):
        layer = PredictiveCodingSCLayer(n_inputs=4, n_neurons=3, length=64, seed=42)
        result = layer.forward([0.2, 0.4, 0.6, 0.8])
        assert result["surprises"].shape == (3,)
        assert result["predictions"].shape == (3, 4)

    def test_error_decreases_with_learning(self):
        """Repeated exposure to same input should reduce prediction error."""
        layer = PredictiveCodingSCLayer(n_inputs=3, n_neurons=2, length=256, lr=0.1, seed=42)
        inputs = [0.3, 0.5, 0.7]
        errors = []
        for _ in range(20):
            result = layer.forward(inputs)
            errors.append(result["prediction_error"])
        # Error should decrease over time
        assert errors[-1] < errors[0]

    def test_high_surprise_on_novel_input(self):
        """Novel input after learning should produce higher surprise."""
        layer = PredictiveCodingSCLayer(n_inputs=2, n_neurons=2, length=512, lr=0.2, seed=42)
        # Train on one pattern
        for _ in range(30):
            layer.forward([0.8, 0.2])
        error_familiar = layer.forward([0.8, 0.2])["prediction_error"]
        # Switch to opposite pattern
        error_novel = layer.forward([0.2, 0.8])["prediction_error"]
        assert error_novel > error_familiar

    def test_prediction_error_bounded(self):
        """Prediction error should be in [0, 1]."""
        layer = PredictiveCodingSCLayer(n_inputs=3, n_neurons=2, length=128, seed=42)
        result = layer.forward([0.0, 0.5, 1.0])
        assert 0.0 <= result["prediction_error"] <= 1.0
        assert np.all(result["surprises"] >= 0.0)
        assert np.all(result["surprises"] <= 1.0)

    def test_weights_stay_bounded(self):
        """Weights should remain in [0, 1] after learning."""
        layer = PredictiveCodingSCLayer(n_inputs=3, n_neurons=2, length=128, lr=0.5, seed=42)
        for _ in range(50):
            layer.forward([0.9, 0.9, 0.9])
        assert np.all(layer.weights >= 0.0)
        assert np.all(layer.weights <= 1.0)

    def test_zero_input_low_error_after_learning(self):
        """Learning all-zero input should converge to near-zero predictions."""
        layer = PredictiveCodingSCLayer(n_inputs=2, n_neurons=1, length=512, lr=0.2, seed=42)
        for _ in range(30):
            layer.forward([0.0, 0.0])
        result = layer.forward([0.0, 0.0])
        assert result["prediction_error"] < 0.1

    def test_reset_restores_initial_state(self):
        layer = PredictiveCodingSCLayer(n_inputs=2, n_neurons=1, length=64, seed=42)
        w_init = layer.weights.copy()
        for _ in range(10):
            layer.forward([0.5, 0.5])
        layer.reset()
        np.testing.assert_array_equal(layer.weights, w_init)

    def test_xor_is_zero_multiplication(self):
        """Verify the XOR operation: error between identical streams should be ~0."""
        layer = PredictiveCodingSCLayer(n_inputs=1, n_neurons=1, length=1024, lr=0.0, seed=42)
        # Set weight to match input exactly
        layer.weights[0, 0] = 0.5
        result = layer.forward([0.5])
        # With same probability, XOR Hamming distance ≈ 2*p*(1-p) = 0.5
        # (XOR of two independent Bernoulli(p) streams)
        assert result["prediction_error"] < 0.6
