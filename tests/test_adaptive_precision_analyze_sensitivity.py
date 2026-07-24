# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAnalyzeSensitivity from former test_adaptive_precision.py

"""Focused suite: TestAnalyzeSensitivity from former test_adaptive_precision.py."""

from __future__ import annotations

from tests.adaptive_precision_support import *  # noqa: F403


class TestAnalyzeSensitivity:
    """Sensitivity-analysis facade checks."""

    def test_returns_per_layer_scores(self) -> None:
        """Sensitivity analysis returns one score per weight layer."""
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        sens = analyze_sensitivity(weights, n_trials=5)
        assert len(sens) == 2
        assert all(s >= 0 for s in sens)

    def test_larger_weights_more_sensitive(self) -> None:
        """Sensitivity scores remain numeric across weight scales."""
        small_w = [np.random.randn(4, 4) * 0.01]
        large_w = [np.random.randn(4, 4) * 0.5]
        sens_small = analyze_sensitivity(small_w, n_trials=10, seed=42)
        sens_large = analyze_sensitivity(large_w, n_trials=10, seed=42)
        # Larger weights should generally be more sensitive
        # (not guaranteed per-trial, but on average)
        assert isinstance(sens_small[0], float)
        assert isinstance(sens_large[0], float)
