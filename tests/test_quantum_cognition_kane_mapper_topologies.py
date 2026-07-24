# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kane silicon mapper topology + coupling contracts

"""Real topology and exchange-coupling contracts for KaneSiliconMapper.

Covers the linear / grid / triangular / hexagonal placement kernels and the
Kane (1998) exponential J(d) model without mocking physical constants.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.kane_mapper import (
    KaneRegisterLayout,
    KaneSiliconMapper,
    _BOHR_RADIUS_STAR_NM,
    _J0_MEV,
)


@pytest.mark.parametrize("topology", ["linear", "grid", "triangular", "hexagonal"])
def test_map_pool_produces_consistent_layout(topology: str) -> None:
    mapper = KaneSiliconMapper(spacing_nm=20.0, topology=topology)
    layout = mapper.map_pool_to_register(n_sites=6)
    assert isinstance(layout, KaneRegisterLayout)
    assert layout.n_qubits == 6
    assert layout.qubit_positions.shape == (6, 2)
    assert layout.coupling_matrix.shape == (6, 6)
    # Coupling is symmetric and zero on the diagonal.
    np.testing.assert_allclose(layout.coupling_matrix, layout.coupling_matrix.T)
    assert np.allclose(np.diag(layout.coupling_matrix), 0.0)
    assert layout.max_gate_depth > 0
    assert layout.t2_budget_ms > 0.0
    assert isinstance(layout.gate_schedule, list)
    assert len(layout.gate_schedule) >= 1
    serialised = layout.to_dict()
    assert serialised["n_qubits"] == 6
    assert len(serialised["qubit_positions_nm"]) == 6


def test_linear_topology_spacing_matches_configuration() -> None:
    spacing = 25.0
    mapper = KaneSiliconMapper(spacing_nm=spacing, topology="linear")
    layout = mapper.map_pool_to_register(4)
    xs = layout.qubit_positions[:, 0]
    diffs = np.diff(xs)
    assert np.allclose(diffs, spacing)
    assert np.allclose(layout.qubit_positions[:, 1], 0.0)


def test_grid_topology_fills_rows() -> None:
    mapper = KaneSiliconMapper(spacing_nm=10.0, topology="grid")
    layout = mapper.map_pool_to_register(4)
    # 2×2 grid at 10 nm.
    expected = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float64)
    # Order is row-major by construction of the mapper.
    np.testing.assert_allclose(layout.qubit_positions, expected)


def test_triangular_topology_staggers_alternate_rows() -> None:
    mapper = KaneSiliconMapper(spacing_nm=10.0, topology="triangular")
    layout = mapper.map_pool_to_register(4)
    # Second row (indices 2,3 with cols=2) must be x-offset by half spacing.
    assert layout.qubit_positions[2, 0] == pytest.approx(5.0)
    assert layout.qubit_positions[2, 1] == pytest.approx(10.0 * math.sqrt(3) / 2)


def test_hexagonal_topology_places_unique_sites() -> None:
    mapper = KaneSiliconMapper(spacing_nm=12.0, topology="hexagonal")
    layout = mapper.map_pool_to_register(5)
    # All five positions must be distinct.
    unique = {tuple(np.round(p, 9)) for p in layout.qubit_positions}
    assert len(unique) == 5


def test_exchange_coupling_matches_kane_formula() -> None:
    d = 20.0
    expected = _J0_MEV * math.exp(-2.0 * d / _BOHR_RADIUS_STAR_NM)
    assert KaneSiliconMapper._exchange_coupling(d) == pytest.approx(expected)
    # Zero / negative distance returns the prefactor (coincident donors).
    assert KaneSiliconMapper._exchange_coupling(0.0) == pytest.approx(_J0_MEV)
    assert KaneSiliconMapper._exchange_coupling(-1.0) == pytest.approx(_J0_MEV)


def test_mapper_validation() -> None:
    with pytest.raises(ValueError, match="spacing_nm"):
        KaneSiliconMapper(spacing_nm=0.0)
    with pytest.raises(ValueError, match="topology"):
        KaneSiliconMapper(topology="ring")
    mapper = KaneSiliconMapper()
    with pytest.raises(ValueError, match="n_sites"):
        mapper.map_pool_to_register(0)


def test_nearest_neighbour_coupling_dominates_far_pairs() -> None:
    mapper = KaneSiliconMapper(spacing_nm=20.0, topology="linear")
    layout = mapper.map_pool_to_register(4)
    nn = layout.coupling_matrix[0, 1]
    far = layout.coupling_matrix[0, 3]
    assert nn > 0.0
    assert far > 0.0
    assert nn > far * 10.0
