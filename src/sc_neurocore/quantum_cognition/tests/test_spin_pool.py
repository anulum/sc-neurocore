# SPDX-License-Identifier: AGPL-3.0-or-later
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

"""Inline tests for SpinPoolMPS."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.spin_pool import SpinPoolMPS


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
