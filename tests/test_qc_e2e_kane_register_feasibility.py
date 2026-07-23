# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKaneRegisterFeasibility from former test_qc_e2e.py

"""Focused suite: TestKaneRegisterFeasibility from former test_qc_e2e.py."""

from __future__ import annotations

from tests.qc_e2e_support import *  # noqa: F403

class TestKaneRegisterFeasibility:
    """Generate large registers, verify properties."""

    def test_512_qubit_grid(self) -> None:
        mapper = KaneSiliconMapper(spacing_nm=10.0, topology="grid")
        layout = mapper.map_pool_to_register(512)
        assert layout.n_qubits == 512
        assert layout.coupling_matrix.shape == (512, 512)
        # Symmetry
        np.testing.assert_array_almost_equal(
            layout.coupling_matrix, layout.coupling_matrix.T, decimal=15
        )
        # All non-negative
        assert np.all(layout.coupling_matrix >= 0.0)
        # Diagonal zero
        np.testing.assert_array_almost_equal(np.diag(layout.coupling_matrix), 0.0)

    def test_constraints_parametric(self) -> None:
        """Verify feasibility transitions at expected spacing."""
        mapper = KaneSiliconMapper(spacing_nm=10.0)
        c10 = mapper.get_constraints(8)
        assert c10["feasible"] is True

        mapper2 = KaneSiliconMapper(spacing_nm=50.0)
        c50 = mapper2.get_constraints(8)
        assert c50["feasible"] is False
