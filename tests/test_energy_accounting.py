# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Tests for sc_neurocore.energy_accounting
from __future__ import annotations
import pytest
from sc_neurocore.energy_accounting import EnergyAccountant, HardwareCostModel
from sc_neurocore.energy_accounting.accountant import HARDWARE_COSTS


class TestHardwareCosts:
    def test_builtin_targets(self):
        assert len(HARDWARE_COSTS) >= 5
        assert "loihi2" in HARDWARE_COSTS
        assert "akida" in HARDWARE_COSTS
        assert "analog_28nm" in HARDWARE_COSTS


class TestEnergyAccountant:
    def test_basic(self):
        acc = EnergyAccountant("loihi2")
        r = acc.account(["h1", "out"], [(64, 32), (32, 10)], [100, 30], n_timesteps=50)
        assert r.total_energy_nj > 0
        assert len(r.layers) == 2

    def test_dominant_layer(self):
        acc = EnergyAccountant("loihi2")
        r = acc.account(["h1", "out"], [(64, 32), (32, 10)], [1000, 10], n_timesteps=50)
        assert r.dominant_layer == "h1"

    def test_energy_per_spike(self):
        acc = EnergyAccountant("loihi2")
        r = acc.account(["h"], [(10, 5)], [100], n_timesteps=20)
        assert r.energy_per_spike_pj > 0

    def test_no_spikes(self):
        acc = EnergyAccountant("loihi2")
        r = acc.account(["h"], [(10, 5)], [0], n_timesteps=10)
        assert r.total_energy_pj > 0  # still membrane updates
        assert r.energy_per_spike_pj == 0.0

    def test_different_hardware(self):
        r_loihi = EnergyAccountant("loihi2").account(["h"], [(10, 5)], [100], 10)
        r_analog = EnergyAccountant("analog_28nm").account(["h"], [(10, 5)], [100], 10)
        assert r_analog.total_energy_pj < r_loihi.total_energy_pj

    def test_custom_cost_model(self):
        custom = HardwareCostModel(name="custom", synop_pj=1.0, membrane_update_pj=0.1)
        acc = EnergyAccountant(custom)
        r = acc.account(["h"], [(4, 2)], [10], 5)
        assert r.hardware == "custom"

    def test_unknown_hardware(self):
        with pytest.raises(ValueError):
            EnergyAccountant("nonexistent")

    def test_summary(self):
        acc = EnergyAccountant("loihi2")
        r = acc.account(["h1", "out"], [(10, 5), (5, 2)], [50, 10], 20)
        s = r.summary()
        assert "loihi2" in s
        assert "nJ" in s

    def test_routing_energy(self):
        acc = EnergyAccountant("loihi2")
        r = acc.account(["h"], [(10, 5)], [100], 10)
        assert r.routing_energy_pj > 0
