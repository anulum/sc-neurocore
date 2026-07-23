# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKnmMatrix from former test_scpn_integrated.py

"""Focused suite: TestKnmMatrix from former test_scpn_integrated.py."""

from __future__ import annotations

from tests.scpn_integrated_support import *  # noqa: F403

class TestKnmMatrix:
    def test_shape(self):
        K = build_knm_matrix(16)
        assert K.shape == (16, 16)

    def test_symmetric(self):
        K = build_knm_matrix(16)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_zero_diagonal(self):
        K = build_knm_matrix(16)
        np.testing.assert_allclose(np.diag(K), 0.0)

    def test_non_negative(self):
        K = build_knm_matrix(16)
        assert np.all(K >= 0), "coupling matrix has negative entries"

    def test_custom_size(self):
        K = build_knm_matrix(8)
        assert K.shape == (8, 8)

    @pytest.mark.parametrize("n_layers", [0, -1, 1.5, True])
    def test_rejects_invalid_layer_count(self, n_layers):
        with pytest.raises(ValueError, match="n_layers"):
            build_knm_matrix(n_layers)

    def test_adjacent_layers_coupled(self):
        """Adjacent layers should have nonzero coupling."""
        K = build_knm_matrix(16)
        for i in range(15):
            assert K[i, i + 1] > 0, f"L{i + 1}↔L{i + 2} not coupled"
