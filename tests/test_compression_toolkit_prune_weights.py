# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPruneWeights from former test_compression_toolkit.py

"""Focused suite: TestPruneWeights from former test_compression_toolkit.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.compression import (
    prune_weights,
    PruningReport,
)


class TestPruneWeights:
    """Magnitude weight pruning."""

    def test_magnitude_pruning(self) -> None:
        """Small weights are zeroed and large ones kept."""
        w = [np.array([[0.5, 0.001, -0.8], [-0.002, 0.3, 0.0]])]
        pruned, report = prune_weights(w, threshold=0.01)
        assert pruned[0][0, 1] == 0.0
        assert pruned[0][1, 0] == 0.0
        assert pruned[0][0, 0] == 0.5

    def test_sparsity_calculation(self) -> None:
        """Two of four weights pruned are reported."""
        w = [np.array([[0.1, 0.001], [0.002, 0.5]])]
        _, report = prune_weights(w, threshold=0.01)
        assert isinstance(report, PruningReport)
        assert report.original_params == 4
        assert report.pruned_params == 2
        assert report.remaining_params == 2
        assert report.sparsity == pytest.approx(0.5)

    def test_percentile_pruning(self) -> None:
        """Percentile pruning reaches its sparsity."""
        rng = np.random.RandomState(42)
        w = [rng.randn(10, 10)]
        pruned, report = prune_weights(w, threshold=50.0, method="percentile")
        assert report.sparsity > 0.3

    def test_no_pruning_below_threshold(self) -> None:
        """Weights above the threshold are all kept."""
        w = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        pruned, report = prune_weights(w, threshold=0.01)
        assert report.pruned_params == 0
        np.testing.assert_array_equal(pruned[0], w[0])

    def test_multiple_layers(self) -> None:
        """Each layer is pruned on its own."""
        w = [np.ones((3, 3)) * 0.5, np.ones((2, 3)) * 0.001]
        pruned, report = prune_weights(w, threshold=0.01)
        assert report.original_params == 15
        assert report.pruned_params == 6
        np.testing.assert_array_equal(pruned[0], w[0])
        np.testing.assert_array_equal(pruned[1], np.zeros((2, 3)))

    def test_does_not_modify_original(self) -> None:
        """Pruning leaves the original weights unchanged."""
        w = [np.array([[0.5, 0.001]])]
        original_copy = w[0].copy()
        prune_weights(w, threshold=0.01)
        np.testing.assert_array_equal(w[0], original_copy)
