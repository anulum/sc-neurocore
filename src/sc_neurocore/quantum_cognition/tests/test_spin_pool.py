# SPDX-License-Identifier: AGPL-3.0-or-later
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

"""Inline tests for SpinPoolMPS."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.spin_pool import SpinCouplingTensor, SpinPoolMPS


class TestSpinPoolMPS:
    def test_init_defaults(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        assert pool.n_sites == 8
        assert pool.bond_dim == 16
        assert pool._measurement_count == 0

    def test_entanglement_map_normalised(self) -> None:
        """Entanglement map should sum to 1 after init."""
        pool = SpinPoolMPS(n_sites=10)
        assert abs(np.sum(pool.entanglement_map) - 1.0) < 1e-10

    def test_measurement_updates_map(self) -> None:
        """Measurement at a site should shift entanglement towards that site."""
        pool = SpinPoolMPS(n_sites=8)
        initial = pool.entanglement_map.copy()
        pool.apply_measurement(3, 1.0)
        # Site 3 should have higher entanglement after measurement
        assert pool.entanglement_map[3] > initial[3]
        assert pool._measurement_count == 1

    def test_normalisation_preserved(self) -> None:
        """Entanglement map should remain normalised after measurements."""
        pool = SpinPoolMPS(n_sites=16)
        for i in range(0, 16, 3):
            pool.apply_measurement(i, 1.0)
        assert abs(np.sum(pool.entanglement_map) - 1.0) < 1e-10

    def test_atp_efficiency_range(self) -> None:
        """ATP efficiency is a singlet probability in [0, 1]."""
        pool = SpinPoolMPS(n_sites=8)
        for i in range(8):
            eff = pool.get_local_atp_efficiency(i)
            assert 0.0 <= eff <= 1.0

    def test_invalid_site_index(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        with pytest.raises(IndexError):
            pool.apply_measurement(4, 1.0)
        with pytest.raises(IndexError):
            pool.apply_measurement(-1, 1.0)

    def test_negative_intensity(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        with pytest.raises(ValueError, match="intensity"):
            pool.apply_measurement(0, -0.5)

    def test_reset(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        pool.apply_measurement(0, 1.0)
        pool.apply_measurement(1, 1.0)
        pool.reset()
        assert pool._measurement_count == 0
        assert abs(np.sum(pool.entanglement_map) - 1.0) < 1e-10

    def test_state_roundtrip(self) -> None:
        """get_state → set_state should preserve state."""
        pool = SpinPoolMPS(n_sites=8)
        pool.apply_measurement(2, 1.0)
        pool.apply_measurement(5, 0.5)
        state = pool.get_state()

        pool2 = SpinPoolMPS(n_sites=8)
        pool2.set_state(state)
        np.testing.assert_array_almost_equal(pool.entanglement_map, pool2.entanglement_map)
        assert pool2._measurement_count == 2

    def test_scpn_payload(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        payload = pool.to_scpn_payload()
        assert "quantum_cognition_spin_pool" in payload
        qc = payload["quantum_cognition_spin_pool"]
        assert qc["n_sites"] == 4
        assert len(qc["entanglement_map"]) == 4
        assert len(qc["atp_efficiencies"]) == 4

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            SpinPoolMPS(n_sites=0)
        with pytest.raises(ValueError):
            SpinPoolMPS(n_sites=4, bond_dim=0)
        with pytest.raises(ValueError):
            SpinPoolMPS(n_sites=4, correlation_length=0.0)
        with pytest.raises(ValueError):
            SpinPoolMPS(n_sites=4, update_rate=0.0)

    def test_singlet_statevector_roundtrip_preserves_atp_observable(self) -> None:
        pool = SpinPoolMPS(n_sites=2, bond_dim=2)
        singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.complex128) / np.sqrt(2.0)

        pool.set_statevector(singlet)

        restored = pool.to_statevector()
        assert abs(np.vdot(singlet, restored)) == pytest.approx(1.0)
        assert pool.get_local_atp_efficiency(0) == pytest.approx(1.0)
        assert pool.get_local_atp_efficiency(1) == pytest.approx(1.0)
        np.testing.assert_allclose(np.trace(pool.rho), 1.0)
        np.testing.assert_allclose(np.diag(pool.rho).real, [0.5, 0.5], atol=1e-12)

    def test_statevector_import_rejects_invalid_public_contracts(self) -> None:
        pool = SpinPoolMPS(n_sites=3, bond_dim=2)

        with pytest.raises(ValueError, match="statevector length"):
            pool.set_statevector(np.ones(4, dtype=np.complex128))
        with pytest.raises(ValueError, match="zero norm"):
            pool.set_statevector(np.zeros(8, dtype=np.complex128))
        with pytest.raises(ValueError, match="Exact statevector export limited"):
            pool.to_statevector(max_sites=2)

    def test_statevector_import_rejects_silent_bond_truncation(self) -> None:
        pool = SpinPoolMPS(n_sites=2, bond_dim=1)
        singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.complex128) / np.sqrt(2.0)

        with pytest.raises(ValueError, match="State requires bond dimension 2"):
            pool.set_statevector(singlet)

    def test_checkpoint_without_map_recomputes_quantum_diagnostics(self) -> None:
        source = SpinPoolMPS(n_sites=2, bond_dim=2)
        bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        source.set_statevector(bell)
        checkpoint = source.get_state()
        checkpoint.pop("entanglement_map")

        restored = SpinPoolMPS(n_sites=2, bond_dim=2)
        restored.set_state(checkpoint)

        assert np.sum(restored.entanglement_map) == pytest.approx(1.0)
        np.testing.assert_allclose(restored.entanglement_map, [0.5, 0.5], atol=1e-12)

    def test_corrupted_checkpoint_zero_norm_fails_on_statevector_export(self) -> None:
        pool = SpinPoolMPS(n_sites=2)
        pool.set_state(
            {
                "tensors": [
                    [[[0.0], [0.0]]],
                    [[[0.0], [0.0]]],
                ],
            }
        )

        with pytest.raises(ValueError, match="MPS state has zero norm"):
            pool.to_statevector()

    def test_single_site_pool_rejects_two_site_atp_observable(self) -> None:
        pool = SpinPoolMPS(n_sites=1)

        with pytest.raises(IndexError, match="Two-site RDM requires"):
            pool.get_local_atp_efficiency(0)

    def test_exact_evolution_rejects_invalid_coupling_contracts(self) -> None:
        pool = SpinPoolMPS(n_sites=3, bond_dim=2)
        tensor = np.eye(3, dtype=np.float64)

        with pytest.raises(ValueError, match="time_us"):
            pool.evolve_exact([], time_us=-1.0)
        with pytest.raises(ValueError, match="Exact dense Hamiltonian evolution limited"):
            pool.evolve_exact([], time_us=0.0, max_sites=2)
        with pytest.raises(IndexError, match="out of range"):
            pool.evolve_exact([SpinCouplingTensor(0, 3, tensor)], time_us=0.0)
        with pytest.raises(ValueError, match="shape"):
            pool.evolve_exact([SpinCouplingTensor(0, 1, np.eye(2))], time_us=0.0)

    def test_internal_heisenberg_same_site_is_noop(self) -> None:
        pool = SpinPoolMPS(n_sites=3, bond_dim=4)
        initial = pool.to_statevector()

        pool._apply_heisenberg_between(1, 1, coupling=0.5)

        np.testing.assert_allclose(pool.to_statevector(), initial, atol=1e-12)

    def test_internal_heisenberg_adjacent_gate_preserves_state_norm(self) -> None:
        pool = SpinPoolMPS(n_sites=2, bond_dim=4)
        initial = np.array([1.0, 1.0j, -0.25, 0.5], dtype=np.complex128)
        pool.set_statevector(initial)

        pool._apply_heisenberg_between(0, 1, coupling=0.3)

        evolved = pool.to_statevector()
        assert np.linalg.norm(evolved) == pytest.approx(1.0)
        assert np.all(np.isfinite(evolved))
        assert not np.allclose(evolved, initial / np.linalg.norm(initial))

    def test_internal_heisenberg_nonadjacent_swap_network_preserves_state_norm(self) -> None:
        pool = SpinPoolMPS(n_sites=3, bond_dim=4)
        initial = np.zeros(8, dtype=np.complex128)
        initial[1] = 1.0 / np.sqrt(2.0)
        initial[6] = 1.0j / np.sqrt(2.0)
        pool.set_statevector(initial)

        pool._apply_heisenberg_between(2, 0, coupling=0.25)

        evolved = pool.to_statevector()
        assert np.linalg.norm(evolved) == pytest.approx(1.0)
        assert np.all(np.isfinite(evolved))
        assert not np.allclose(evolved, initial)
