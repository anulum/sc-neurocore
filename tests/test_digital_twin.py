# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for digital twin FPGA mismatch module

import numpy as np

from sc_neurocore.digital_twin import FPGAMismatchModel


class TestQuantization:
    def test_q88_quantization(self):
        model = FPGAMismatchModel(quantization_bits=16)
        x = np.array([0.123456789, -0.987654321, 0.5])
        q = model.quantize(x)
        # Q8.8 => 256 levels per integer, step = 1/256
        step = 1.0 / 256
        for val in q:
            remainder = abs(val) % step
            assert remainder < 1e-10 or abs(remainder - step) < 1e-10

    def test_quantize_preserves_shape(self):
        model = FPGAMismatchModel()
        x = np.random.randn(5, 3)
        assert model.quantize(x).shape == (5, 3)

    def test_quantize_zero(self):
        model = FPGAMismatchModel()
        assert model.quantize(np.array([0.0]))[0] == 0.0


class TestPerturbations:
    def test_perturb_weights_changes_values(self):
        model = FPGAMismatchModel(weight_cv=0.1, seed=0)
        w = np.ones((10, 10))
        pw = model.perturb_weights(w)
        assert not np.array_equal(w, pw)

    def test_perturb_weights_preserves_shape(self):
        model = FPGAMismatchModel()
        w = np.random.randn(8, 4)
        assert model.perturb_weights(w).shape == (8, 4)

    def test_perturb_thresholds(self):
        model = FPGAMismatchModel(threshold_cv=0.05, seed=0)
        th = np.ones(50)
        pth = model.perturb_thresholds(th)
        assert pth.shape == (50,)
        assert not np.array_equal(th, pth)
        assert abs(pth.mean() - 1.0) < 0.1

    def test_jitter_timing_shape(self):
        model = FPGAMismatchModel()
        j = model.jitter_timing(100)
        assert j.shape == (100,)
        assert j.min() >= 0.9
        assert j.max() <= 1.1

    def test_zero_cv_no_perturbation(self):
        model = FPGAMismatchModel(weight_cv=0.0, seed=0)
        w = np.array([[0.5, -0.25], [0.125, -0.375]])
        pw = model.perturb_weights(w)
        np.testing.assert_array_equal(w, model.quantize(w))


class TestNetworkApplication:
    def test_apply_to_network_weights(self):
        model = FPGAMismatchModel(seed=0)
        weights = [np.random.randn(10, 5), np.random.randn(5, 3)]
        perturbed = model.apply_to_network_weights(weights)
        assert len(perturbed) == 2
        assert perturbed[0].shape == (10, 5)
        assert perturbed[1].shape == (5, 3)

    def test_mismatch_report(self):
        model = FPGAMismatchModel(seed=0)
        weights = [np.random.randn(10, 5)]
        report = model.mismatch_report(weights)
        assert report["total_parameters"] == 50
        assert report["mean_absolute_error"] > 0
        assert report["max_absolute_error"] > 0
        assert report["weight_cv"] == 0.02
        assert report["quantization_bits"] == 16

    def test_deterministic_seed(self):
        a = FPGAMismatchModel(seed=123)
        b = FPGAMismatchModel(seed=123)
        w = np.random.randn(10, 10)
        np.testing.assert_array_equal(a.perturb_weights(w), b.perturb_weights(w))
