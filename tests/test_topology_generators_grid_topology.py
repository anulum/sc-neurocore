# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGridTopology from former test_topology_generators.py

"""Focused suite: TestGridTopology from former test_topology_generators.py."""

from __future__ import annotations

from tests.topology_generators_support import *  # noqa: F403


class TestGridTopology:
    def test_csr_valid(self) -> None:
        indptr, indices, data = grid_topology(5, 5, radius=1, weight=1.0)
        _validate_csr(indptr, indices, data, 25, 25)

    def test_corner_degree(self) -> None:
        """Corner neuron in 5x5 grid, r=1 has 3 neighbours."""
        indptr, indices, data = grid_topology(5, 5, radius=1, weight=1.0)
        mat = _csr_to_dense(indptr, indices, data, 25, 25)
        corner_deg = (mat[0] != 0).sum()
        assert corner_deg == 3, f"corner degree {corner_deg} != 3"

    def test_centre_degree(self) -> None:
        """Centre neuron in 5x5 grid, r=1 has 8 neighbours."""
        indptr, indices, data = grid_topology(5, 5, radius=1, weight=1.0)
        mat = _csr_to_dense(indptr, indices, data, 25, 25)
        centre = 2 * 5 + 2  # row 2, col 2
        centre_deg = (mat[centre] != 0).sum()
        assert centre_deg == 8, f"centre degree {centre_deg} != 8"

    def test_no_self_connections(self) -> None:
        indptr, indices, data = grid_topology(4, 4, radius=1, weight=1.0)
        mat = _csr_to_dense(indptr, indices, data, 16, 16)
        assert np.all(np.diag(mat) == 0)
