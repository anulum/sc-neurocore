# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerturbations from former test_digital_twin.py

"""Focused suite: TestPerturbations from former test_digital_twin.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.digital_twin import (
    FPGAMismatchModel,
)


class TestPerturbations:
    """Device mismatch perturbations."""

    def test_perturb_weights_changes_values(self) -> None:
        """Perturbation changes the weights."""
        model = FPGAMismatchModel(weight_cv=0.1, seed=0)
        w = np.ones((10, 10))
        pw = model.perturb_weights(w)
        assert not np.array_equal(w, pw)

    def test_perturb_weights_preserves_shape(self) -> None:
        """Perturbation keeps the weight shape."""
        model = FPGAMismatchModel()
        w = np.random.randn(8, 4)
        assert model.perturb_weights(w).shape == (8, 4)

    def test_perturb_thresholds(self) -> None:
        """Perturbed thresholds stay near their mean."""
        model = FPGAMismatchModel(threshold_cv=0.05, seed=0)
        th = np.ones(50)
        pth = model.perturb_thresholds(th)
        assert pth.shape == (50,)
        assert not np.array_equal(th, pth)
        assert abs(pth.mean() - 1.0) < 0.1

    def test_jitter_timing_shape(self) -> None:
        """Timing jitter stays within ten percent."""
        model = FPGAMismatchModel()
        j = model.jitter_timing(100)
        assert j.shape == (100,)
        assert j.min() >= 0.9
        assert j.max() <= 1.1

    def test_zero_cv_no_perturbation(self) -> None:
        """Zero variation only quantises the weights."""
        model = FPGAMismatchModel(weight_cv=0.0, seed=0)
        w = np.array([[0.5, -0.25], [0.125, -0.375]])
        pw = model.perturb_weights(w)
        np.testing.assert_array_equal(w, model.quantize(w))
