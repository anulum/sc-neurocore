# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.energy_accounting
from __future__ import annotations
import pytest
from sc_neurocore.energy_accounting import EnergyAccountant, HardwareCostModel
from sc_neurocore.energy_accounting.accountant import HARDWARE_COSTS, EnergyReport
from sc_neurocore.energy_accounting.unified_reporter import (
    UnifiedEnergyReport,
    UnifiedEnergyReporter,
)


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

    def test_dominant_layer_empty(self):
        r = EnergyReport(hardware="loihi2")
        assert r.dominant_layer is None

    def test_energy_per_spike_matches_total_energy_ratio(self):
        acc = EnergyAccountant("loihi2")
        r = acc.account(["h1", "h2"], [(8, 4), (4, 2)], [40, 20], n_timesteps=10)
        expected = r.total_energy_pj / (40 + 20)
        assert r.energy_per_spike_pj == pytest.approx(expected)


class TestUnifiedEnergyReporter:
    def test_summary_includes_asic_line_conditionally(self):
        no_asic = UnifiedEnergyReport(total_power_mw=5.0, carbon_g_co2=0.01, junction_temp_c=30.0)
        with_asic = UnifiedEnergyReport(
            total_power_mw=7.0,
            carbon_g_co2=0.02,
            junction_temp_c=31.0,
            asic_power_mw=2.0,
        )
        assert "ASIC power" not in no_asic.summary()
        assert "ASIC power" in with_asic.summary()

    def test_analyse_adds_layer_and_asic_power(self):
        reporter = UnifiedEnergyReporter(asic_power_mw=3.0)
        report = reporter.analyze(
            layer_configs=[{"power_mw": 2.0}, {"power_mw": 1.0}],
            total_power_mw=4.0,
            inference_time_s=0.5,
        )
        assert report.total_power_mw == pytest.approx(10.0)
        assert report.grid_region
        assert isinstance(report.thermal_safe, bool)

    def test_analyse_without_layer_configs_uses_total_and_asic_only(self):
        reporter = UnifiedEnergyReporter(asic_power_mw=2.5)
        report = reporter.analyze(
            layer_configs=None,
            total_power_mw=5.5,
            inference_time_s=1.0,
        )
        assert report.total_power_mw == pytest.approx(8.0)
        assert report.asic_power_mw == pytest.approx(2.5)
        assert report.grid_region
