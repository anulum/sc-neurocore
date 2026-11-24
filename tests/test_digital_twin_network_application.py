# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNetworkApplication from former test_digital_twin.py

"""Focused suite: TestNetworkApplication from former test_digital_twin.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.digital_twin import (
    FPGAMismatchModel,
)


class TestNetworkApplication:
    """Applying FPGA mismatch to a whole network."""

    def test_apply_to_network_weights(self) -> None:
        """Each layer is perturbed and keeps its shape."""
        model = FPGAMismatchModel(seed=0)
        weights = [np.random.randn(10, 5), np.random.randn(5, 3)]
        perturbed = model.apply_to_network_weights(weights)
        assert len(perturbed) == 2
        assert perturbed[0].shape == (10, 5)
        assert perturbed[1].shape == (5, 3)

    def test_mismatch_report(self) -> None:
        """The report counts parameters and positive errors."""
        model = FPGAMismatchModel(seed=0)
        weights = [np.random.randn(10, 5)]
        report = model.mismatch_report(weights)
        assert report["total_parameters"] == 50
        mean_error = report["mean_absolute_error"]
        max_error = report["max_absolute_error"]
        assert isinstance(mean_error, float)
        assert isinstance(max_error, float)
        assert mean_error > 0
        assert max_error > 0
        assert report["weight_cv"] == 0.02
        assert report["quantization_bits"] == 16

    def test_deterministic_seed(self) -> None:
        """The same seed perturbs the same way."""
        a = FPGAMismatchModel(seed=123)
        b = FPGAMismatchModel(seed=123)
        w = np.random.randn(10, 10)
        np.testing.assert_array_equal(a.perturb_weights(w), b.perturb_weights(w))
