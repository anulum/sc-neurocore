# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact MPS / Heisenberg path coverage for SpinPoolMPS

"""Exact MPS, statevector, and Heisenberg TEBD contracts for SpinPoolMPS.

These paths were under-exercised by the snapshot-style suite; they exercise
real linear-algebra (SVD, statevector load/export, adjacent and SWAP-network
two-site gates) rather than mock handles.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.spin_pool import SpinCouplingTensor, SpinPoolMPS


def test_to_statevector_rejects_oversized_systems() -> None:
    pool = SpinPoolMPS(n_sites=8, bond_dim=4)
    with pytest.raises(ValueError, match="Exact statevector export limited"):
        pool.to_statevector(max_sites=4)


def test_to_statevector_product_state_is_all_up() -> None:
    pool = SpinPoolMPS(n_sites=3, bond_dim=4)
    vec = pool.to_statevector()
    assert vec.shape == (8,)
    assert np.isclose(np.linalg.norm(vec), 1.0)
    # Product |000⟩ has amplitude 1 on the first computational basis state.
    assert np.isclose(abs(vec[0]), 1.0)
    assert np.allclose(vec[1:], 0.0)


def test_set_statevector_roundtrip_preserves_bell_pair() -> None:
    """Load a two-qubit Bell pair and recover it through the MPS bond."""
    pool = SpinPoolMPS(n_sites=2, bond_dim=4)
    # |Φ+⟩ = (|00⟩ + |11⟩) / √2
    phi_plus = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2.0)
    pool.set_statevector(phi_plus)
    recovered = pool.to_statevector()
    # Global phase free comparison.
    overlap = abs(np.vdot(phi_plus, recovered))
    assert overlap == pytest.approx(1.0, abs=1e-10)


def test_set_statevector_rejects_wrong_length_and_zero_norm() -> None:
    pool = SpinPoolMPS(n_sites=2, bond_dim=4)
    with pytest.raises(ValueError, match="statevector length"):
        pool.set_statevector(np.ones(3, dtype=np.complex128))
    with pytest.raises(ValueError, match="zero norm"):
        pool.set_statevector(np.zeros(4, dtype=np.complex128))


def test_set_statevector_rejects_bond_truncation() -> None:
    """A highly entangled four-site state cannot fit in bond_dim=1."""
    pool = SpinPoolMPS(n_sites=4, bond_dim=1)
    # Haar-ish random statevector (seeded) — typically full Schmidt rank.
    rng = np.random.default_rng(0)
    vec = rng.normal(size=16) + 1j * rng.normal(size=16)
    vec = vec / np.linalg.norm(vec)
    with pytest.raises(ValueError, match="bond dimension"):
        pool.set_statevector(vec, atol=1e-14)


def test_evolve_exact_rejects_negative_time_and_oversized() -> None:
    pool = SpinPoolMPS(n_sites=3, bond_dim=8)
    tensor = SpinCouplingTensor(
        i=0,
        j=1,
        tensor_mhz=np.eye(3, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="time_us"):
        pool.evolve_exact([tensor], time_us=-1.0)
    big = SpinPoolMPS(n_sites=8, bond_dim=4)
    with pytest.raises(ValueError, match="max_sites|sites"):
        big.evolve_exact([tensor], time_us=0.1, max_sites=2)


def test_evolve_exact_zero_time_is_identity() -> None:
    pool = SpinPoolMPS(n_sites=2, bond_dim=4)
    before = pool.to_statevector().copy()
    tensor = SpinCouplingTensor(
        i=0,
        j=1,
        tensor_mhz=np.eye(3, dtype=np.float64) * 0.5,
    )
    pool.evolve_exact([tensor], time_us=0.0)
    after = pool.to_statevector()
    assert np.allclose(before, after)


def test_heisenberg_adjacent_gate_preserves_norm() -> None:
    pool = SpinPoolMPS(n_sites=3, bond_dim=8)
    # Seed a non-product state by measuring site 1.
    pool.apply_measurement(1, intensity=1.0)
    before = pool.to_statevector()
    pool._apply_heisenberg_between(0, 1, coupling=0.35)
    after = pool.to_statevector()
    assert np.isclose(np.linalg.norm(after), 1.0, atol=1e-10)
    # Gate should change the state for a non-trivial coupling.
    assert not np.allclose(before, after)


def test_heisenberg_nonadjacent_uses_swap_network() -> None:
    """Non-adjacent Heisenberg must still preserve unitarity (norm=1)."""
    pool = SpinPoolMPS(n_sites=4, bond_dim=8)
    pool.apply_measurement(0, intensity=1.0)
    pool.apply_measurement(3, intensity=0.7)
    before = pool.to_statevector()
    pool._apply_heisenberg_between(0, 3, coupling=0.2)
    after = pool.to_statevector()
    assert np.isclose(np.linalg.norm(after), 1.0, atol=1e-9)
    assert not np.allclose(before, after)


def test_tebd_requires_adjacent_sites() -> None:
    pool = SpinPoolMPS(n_sites=4, bond_dim=4)
    with pytest.raises(ValueError, match="adjacent"):
        pool._apply_tebd_gate(0, 2, coupling=0.1)


def test_rho_property_is_valid_density_matrix() -> None:
    pool = SpinPoolMPS(n_sites=3, bond_dim=4)
    pool.apply_measurement(0, 1.0)
    rho = pool.rho
    assert rho.shape == (2, 2)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-10)
    # Hermitian PSD: eigenvalues non-negative.
    evals = np.linalg.eigvalsh(rho)
    assert np.all(evals >= -1e-12)


def test_set_state_without_entanglement_map_rebuilds_diagnostic() -> None:
    pool = SpinPoolMPS(n_sites=3, bond_dim=4)
    pool.apply_measurement(1, 1.0)
    state = pool.get_state()
    del state["entanglement_map"]
    other = SpinPoolMPS(n_sites=3, bond_dim=4)
    other.set_state(state)
    assert other.entanglement_map.shape == (3,)
    assert np.isclose(np.sum(other.entanglement_map), 1.0)
