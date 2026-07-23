# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpinPoolMPS from former test_quantum_cognition.py

"""Focused suite: TestSpinPoolMPS from former test_quantum_cognition.py."""

from __future__ import annotations

from tests.quantum_cognition_support import *  # noqa: F403

class TestSpinPoolMPS:
    """Tests for the Matrix Product States spin pool emulator."""

    def test_init_defaults(self) -> None:
        pool = SpinPoolMPS()
        assert pool.n_sites == 8
        assert pool.bond_dim == 16
        assert pool.entanglement_map.shape == (8,)
        assert np.isclose(np.sum(pool.entanglement_map), 1.0)

    def test_init_custom(self) -> None:
        pool = SpinPoolMPS(n_sites=4, bond_dim=8, correlation_length=3.0, update_rate=0.2)
        assert pool.n_sites == 4
        assert pool.bond_dim == 8
        assert pool.correlation_length == 3.0
        assert pool.update_rate == 0.2

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="n_sites"):
            SpinPoolMPS(n_sites=0)
        with pytest.raises(ValueError, match="bond_dim"):
            SpinPoolMPS(bond_dim=0)
        with pytest.raises(ValueError, match="correlation_length"):
            SpinPoolMPS(correlation_length=-1.0)
        with pytest.raises(ValueError, match="update_rate"):
            SpinPoolMPS(update_rate=0.0)
        with pytest.raises(ValueError, match="update_rate"):
            SpinPoolMPS(update_rate=1.5)

    def test_apply_measurement_updates_map(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        initial_map = pool.entanglement_map.copy()
        pool.apply_measurement(3, 1.0)
        # After measurement at site 3, entanglement should concentrate near site 3
        assert not np.allclose(pool.entanglement_map, initial_map)
        assert np.isclose(np.sum(pool.entanglement_map), 1.0)
        assert pool._measurement_count == 1

    def test_measurement_site_bounds(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        with pytest.raises(IndexError, match="site_idx"):
            pool.apply_measurement(-1)
        with pytest.raises(IndexError, match="site_idx"):
            pool.apply_measurement(4)
        with pytest.raises(ValueError, match="intensity"):
            pool.apply_measurement(0, intensity=-0.5)

    def test_atp_efficiency_range(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        for i in range(8):
            eff = pool.get_local_atp_efficiency(i)
            assert 0.0 <= eff <= 1.0, f"Efficiency {eff} out of range at site {i}"

    def test_atp_efficiency_site_bounds(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        with pytest.raises(IndexError, match="site_idx"):
            pool.get_local_atp_efficiency(4)

    def test_repeated_measurements_concentrate_entanglement(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        for _ in range(50):
            pool.apply_measurement(0, 1.0)
        # Entanglement should be highest at site 0 after repeated spikes there
        assert pool.entanglement_map[0] > pool.entanglement_map[7]

    def test_get_status(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        pool.apply_measurement(1)
        status = pool.get_status()
        assert status["n_sites"] == 4
        assert status["measurement_count"] == 1
        assert "avg_entanglement" in status
        assert status["coherence_status"] == "stable"

    def test_get_set_state_roundtrip(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        pool.apply_measurement(2, 0.5)
        state = pool.get_state()
        pool2 = SpinPoolMPS(n_sites=4)
        pool2.set_state(state)
        np.testing.assert_array_almost_equal(pool.entanglement_map, pool2.entanglement_map)

    def test_reset(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        pool.apply_measurement(0)
        pool.apply_measurement(1)
        pool.reset()
        assert pool._measurement_count == 0
        assert np.allclose(pool.entanglement_map, 0.25)

    def test_to_scpn_payload(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        payload = pool.to_scpn_payload()
        assert "quantum_cognition_spin_pool" in payload
        inner = payload["quantum_cognition_spin_pool"]
        assert inner["n_sites"] == 4
        assert len(inner["entanglement_map"]) == 4
        assert len(inner["atp_efficiencies"]) == 4

    def test_repr(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        r = repr(pool)
        assert "SpinPoolMPS" in r
        assert "n_sites=4" in r
