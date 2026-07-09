# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.digital_twin

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.digital_twin import FPGAMismatchModel


class TestFPGAMismatchModel:
    def test_defaults(self):
        m = FPGAMismatchModel()
        assert m.quantization_bits == 16
        assert m.weight_cv == 0.02
        assert m.threshold_cv == 0.05
        assert m.clock_jitter_pct == 0.01

    def test_quantize(self):
        m = FPGAMismatchModel(quantization_bits=16)
        v = np.array([0.123456789])
        q = m.quantize(v)
        assert q[0] != v[0]
        assert abs(q[0] - v[0]) < 0.01

    def test_quantize_exact_value(self):
        m = FPGAMismatchModel(quantization_bits=16)
        v = np.array([0.5])
        q = m.quantize(v)
        assert q[0] == pytest.approx(0.5)

    def test_perturb_weights(self):
        m = FPGAMismatchModel(weight_cv=0.02, seed=42)
        w = np.ones((10, 10))
        p = m.perturb_weights(w)
        assert p.shape == w.shape
        assert not np.array_equal(p, w)
        # Perturbation should be small
        assert np.abs(p - w).max() < 0.2

    def test_perturb_thresholds(self):
        m = FPGAMismatchModel(threshold_cv=0.05, seed=42)
        thresholds = np.ones(20)
        p = m.perturb_thresholds(thresholds)
        assert p.shape == thresholds.shape
        assert not np.array_equal(p, thresholds)

    def test_jitter_timing(self):
        m = FPGAMismatchModel(clock_jitter_pct=0.01, seed=42)
        jitter = m.jitter_timing(100)
        assert jitter.shape == (100,)
        assert jitter.min() >= 0.9
        assert jitter.max() <= 1.1
        assert abs(jitter.mean() - 1.0) < 0.05

    def test_apply_to_network_weights(self):
        m = FPGAMismatchModel(seed=42)
        weights = [np.random.randn(5, 5), np.random.randn(3, 5)]
        perturbed = m.apply_to_network_weights(weights)
        assert len(perturbed) == 2
        assert perturbed[0].shape == (5, 5)
        assert perturbed[1].shape == (3, 5)

    def test_mismatch_report(self):
        m = FPGAMismatchModel(seed=42)
        weights = [np.random.randn(5, 5)]
        report = m.mismatch_report(weights)
        assert report["total_parameters"] == 25
        assert report["mean_absolute_error"] > 0
        assert report["max_absolute_error"] > 0
        assert report["weight_cv"] == 0.02
        assert report["quantization_bits"] == 16

    def test_reproducible_with_seed(self):
        m1 = FPGAMismatchModel(seed=123)
        m2 = FPGAMismatchModel(seed=123)
        w = np.random.randn(5, 5)
        np.testing.assert_array_equal(m1.perturb_weights(w), m2.perturb_weights(w))

    def test_different_seeds_different_results(self):
        m1 = FPGAMismatchModel(seed=1)
        m2 = FPGAMismatchModel(seed=2)
        w = np.ones((5, 5))
        p1 = m1.perturb_weights(w)
        p2 = m2.perturb_weights(w)
        assert not np.array_equal(p1, p2)

    def test_zero_cv_no_perturbation(self):
        m = FPGAMismatchModel(weight_cv=0.0, quantization_bits=16, seed=42)
        w = np.array([[0.5]])
        q = m.quantize(w)
        p = m.perturb_weights(w)
        np.testing.assert_array_equal(p, q)
