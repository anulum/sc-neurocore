# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAssignLengths from former test_adaptive_precision.py

"""Focused suite: TestAssignLengths from former test_adaptive_precision.py."""

from __future__ import annotations

from tests.adaptive_precision_support import *  # noqa: F403


class TestAssignLengths:
    """Layer-level bitstream-length assignment checks."""

    def test_hoeffding_produces_assignments(self) -> None:
        """Hoeffding planning returns one assignment per layer."""
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights, method="hoeffding")
        assert len(result) == 2
        assert all(isinstance(r, LayerPrecision) for r in result)

    def test_assignments_respect_bounds(self) -> None:
        """Assigned lengths stay inside caller-provided bounds."""
        weights = [np.random.randn(4, 2)]
        result = assign_lengths(weights, min_length=64, max_length=512)
        assert all(64 <= r.bitstream_length <= 512 for r in result)

    def test_lengths_are_power_of_two(self) -> None:
        """Assigned bitstream lengths are powers of two."""
        weights = [np.random.randn(8, 4), np.random.randn(4, 8)]
        result = assign_lengths(weights, method="hoeffding")
        for r in result:
            L = r.bitstream_length
            assert L & (L - 1) == 0, f"L={L} is not a power of 2"

    def test_relaxed_target_gives_shorter_lengths(self) -> None:
        """Relaxed error targets do not require longer bitstreams."""
        weights = [np.random.randn(4, 2)]
        tight = assign_lengths(weights, target_error=0.01, max_length=4096)
        relaxed = assign_lengths(weights, target_error=0.2, max_length=4096)
        assert relaxed[0].bitstream_length <= tight[0].bitstream_length

    def test_custom_layer_names(self) -> None:
        """Custom layer names propagate into assignments."""
        weights = [np.random.randn(4, 2)]
        result = assign_lengths(weights, layer_names=["my_layer"])
        assert result[0].name == "my_layer"

    def test_sensitivity_path_defaults_budget_to_full_width(self) -> None:
        """Non-Hoeffding planning defaults to a full-width total budget."""
        # A non-Hoeffding method follows the sensitivity-proportional branch;
        # with no explicit budget it defaults to max_length * n_layers and still
        # produces one bounded assignment per layer.
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights, method="proportional", max_length=512)
        assert len(result) == 2
        assert all(r.bitstream_length <= 512 for r in result)
        assert all(r.sensitivity >= 0.0 for r in result)

    def test_default_layer_names(self) -> None:
        """Default layer names are deterministic."""
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights)
        assert result[0].name == "layer_0"
        assert result[1].name == "layer_1"

    def test_sensitivity_method(self) -> None:
        """Sensitivity planning returns non-negative sensitivity scores."""
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights, method="sensitivity", total_budget=2048)
        assert len(result) == 2
        assert all(r.sensitivity >= 0 for r in result)
