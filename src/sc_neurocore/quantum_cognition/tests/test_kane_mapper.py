# SPDX-License-Identifier: AGPL-3.0-or-later
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

"""Inline tests for KaneSiliconMapper."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.kane_mapper import (
    KaneSiliconMapper,
)


class TestKaneSiliconMapper:
    def test_linear_positions(self) -> None:
        """Linear topology should place qubits in a line."""
        mapper = KaneSiliconMapper(spacing_nm=20.0, topology="linear")
        layout = mapper.map_pool_to_register(4)
        assert layout.n_qubits == 4
        assert layout.qubit_positions.shape == (4, 2)
        # All y-coordinates should be zero (1D line)
        assert np.allclose(layout.qubit_positions[:, 1], 0.0)
        # x-spacing should be uniform
        dx = np.diff(layout.qubit_positions[:, 0])
        assert np.allclose(dx, 20.0)

    def test_grid_positions(self) -> None:
        """Grid topology should place qubits in a 2D arrangement."""
        mapper = KaneSiliconMapper(spacing_nm=25.0, topology="grid")
        layout = mapper.map_pool_to_register(9)
        assert layout.n_qubits == 9
        # 9 qubits → 3×3 grid
        assert layout.qubit_positions.shape == (9, 2)

    def test_coupling_matrix_symmetry(self) -> None:
        """Coupling matrix must be symmetric with zero diagonal."""
        mapper = KaneSiliconMapper(spacing_nm=20.0)
        layout = mapper.map_pool_to_register(5)
        J = layout.coupling_matrix
        assert J.shape == (5, 5)
        np.testing.assert_array_almost_equal(J, J.T)
        np.testing.assert_array_almost_equal(np.diag(J), 0.0)

    def test_coupling_decay(self) -> None:
        """Coupling should decay with distance."""
        mapper = KaneSiliconMapper(spacing_nm=20.0, topology="linear")
        layout = mapper.map_pool_to_register(4)
        J = layout.coupling_matrix
        # Nearest neighbours should have stronger coupling than next-nearest
        assert J[0, 1] > J[0, 2] > J[0, 3]

    def test_coupling_positive(self) -> None:
        """All coupling values should be non-negative."""
        mapper = KaneSiliconMapper()
        layout = mapper.map_pool_to_register(8)
        assert np.all(layout.coupling_matrix >= 0.0)

    def test_t2_budget(self) -> None:
        """T₂ budget must be positive."""
        mapper = KaneSiliconMapper()
        layout = mapper.map_pool_to_register(4)
        assert layout.t2_budget_ms > 0
        assert layout.max_gate_depth > 0

    def test_single_qubit(self) -> None:
        """Single qubit register should work."""
        mapper = KaneSiliconMapper()
        layout = mapper.map_pool_to_register(1)
        assert layout.n_qubits == 1
        assert layout.coupling_matrix.shape == (1, 1)
        assert layout.coupling_matrix[0, 0] == 0.0

    def test_constraints(self) -> None:
        """Constraints dict should contain all expected fields."""
        # 10nm spacing is feasible (coupling > 1e-6 meV)
        mapper = KaneSiliconMapper(spacing_nm=10.0)
        c = mapper.get_constraints(8)
        assert "n_sites" in c
        assert "nearest_neighbour_coupling_meV" in c
        assert "feasible" in c
        assert c["feasible"] is True  # 10nm spacing is feasible

    def test_constraints_infeasible(self) -> None:
        """Wide spacing should be infeasible (coupling too weak)."""
        mapper = KaneSiliconMapper(spacing_nm=50.0)
        c = mapper.get_constraints(8)
        assert c["feasible"] is False

    def test_serialisation(self) -> None:
        """to_dict should produce JSON-compatible output."""
        mapper = KaneSiliconMapper()
        layout = mapper.map_pool_to_register(4)
        d = layout.to_dict()
        assert d["n_qubits"] == 4
        assert len(d["qubit_positions_nm"]) == 4
        assert len(d["coupling_matrix_meV"]) == 4

    def test_invalid_spacing(self) -> None:
        with pytest.raises(ValueError, match="spacing_nm"):
            KaneSiliconMapper(spacing_nm=-1)

    def test_invalid_topology(self) -> None:
        with pytest.raises(ValueError, match="topology"):
            KaneSiliconMapper(topology="bcc")

    def test_repr(self) -> None:
        mapper = KaneSiliconMapper(spacing_nm=30.0, topology="grid")
        r = repr(mapper)
        assert "30.0" in r
        assert "grid" in r
