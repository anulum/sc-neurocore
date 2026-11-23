# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKnmMatrix from former test_scpn_integrated.py

"""Focused suite: TestKnmMatrix from former test_scpn_integrated.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.scpn.params import (
    build_knm_matrix,
)


class TestKnmMatrix:
    """The inter-layer coupling matrix."""

    def test_shape(self) -> None:
        """The default matrix is 16 by 16."""
        K = build_knm_matrix(16)
        assert K.shape == (16, 16)

    def test_symmetric(self) -> None:
        """The coupling matrix is symmetric."""
        K = build_knm_matrix(16)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_zero_diagonal(self) -> None:
        """No layer couples to itself."""
        K = build_knm_matrix(16)
        np.testing.assert_allclose(np.diag(K), 0.0)

    def test_non_negative(self) -> None:
        """Every coupling is non-negative."""
        K = build_knm_matrix(16)
        assert np.all(K >= 0), "coupling matrix has negative entries"

    def test_custom_size(self) -> None:
        """A custom layer count sets the size."""
        K = build_knm_matrix(8)
        assert K.shape == (8, 8)

    @pytest.mark.parametrize("n_layers", [0, -1, 1.5, True])
    def test_rejects_invalid_layer_count(self, n_layers: Any) -> None:
        """A layer count that is not a positive integer is rejected."""
        with pytest.raises(ValueError, match="n_layers"):
            build_knm_matrix(n_layers)

    def test_adjacent_layers_coupled(self) -> None:
        """Adjacent layers should have nonzero coupling."""
        K = build_knm_matrix(16)
        for i in range(15):
            assert K[i, i + 1] > 0, f"L{i + 1}↔L{i + 2} not coupled"
