# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPredictiveCodingLayer from former test_predictive_coding.py

"""Focused suite: TestPredictiveCodingLayer from former test_predictive_coding.py."""

from __future__ import annotations

from neuro_symbolic_predictive_coding_support import *  # noqa: F403

class TestPredictiveCodingLayer:
    def test_predict_shape(self):
        layer = PredictiveCodingLayer(input_dim=16, hidden_dim=8)
        pred = layer.predict()
        assert pred.shape == (16,)

    def test_error_shape(self):
        layer = PredictiveCodingLayer(input_dim=16, hidden_dim=8)
        obs = np.random.default_rng(0).normal(0, 0.5, 16).astype(np.float32)
        error = layer.compute_error(obs)
        assert error.shape == (16,)

    def test_update_reduces_error(self):
        rng = np.random.default_rng(42)
        layer = PredictiveCodingLayer(input_dim=8, hidden_dim=4, lr=0.05, seed=42)
        target = rng.normal(0, 0.3, 8).astype(np.float32)
        errors = []
        for _ in range(200):
            mae = layer.update(target)
            errors.append(mae)
        assert errors[-1] < errors[0], "error should decrease over iterations"

    def test_convergence_flag(self):
        layer = PredictiveCodingLayer(input_dim=4, hidden_dim=2, lr=0.05, seed=0)
        target = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        for _ in range(500):
            layer.update(target)
        assert layer.converged or layer.mean_recent_error < 0.1

    def test_precision_scaling(self):
        obs = np.ones(4, dtype=np.float32) * 0.5
        low = PredictiveCodingLayer(4, 2, precision=0.1, seed=0)
        high = PredictiveCodingLayer(4, 2, precision=10.0, seed=0)
        err_low = np.abs(low.compute_error(obs)).mean()
        err_high = np.abs(high.compute_error(obs)).mean()
        assert err_high > err_low, "higher precision should amplify errors"
