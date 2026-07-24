# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNetworkApplication from former test_digital_twin.py

"""Focused suite: TestNetworkApplication from former test_digital_twin.py."""

from __future__ import annotations

from tests.digital_twin_support import *  # noqa: F403


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
