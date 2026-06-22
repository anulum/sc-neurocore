# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sustainability Profiler Tests

import pytest

from sc_neurocore.energy_accounting.sustainability_profiler import (
    CarbonModel,
    EmbodiedCarbon,
    EnergyHarvester,
    EnergyStorageSim,
    FPGAResourceReport,
    GridRegion,
    HarvestProfile,
    MultiHarvestStack,
    SustainabilityOptimizer,
    ThermalModel,
    analyze_multi_harvest,
)


# ── FPGAResourceReport Tests ────────────────────────────────────────


class TestFPGAResourceReport:
    def test_dynamic_power_positive(self):
        r = FPGAResourceReport(luts=10000, ffs=5000, clock_mhz=100)
        assert r.dynamic_power_mw > 0

    def test_dynamic_power_increases_with_luts(self):
        a = FPGAResourceReport(luts=1000)
        b = FPGAResourceReport(luts=100000)
        assert b.dynamic_power_mw > a.dynamic_power_mw

    def test_dynamic_power_increases_with_toggle(self):
        a = FPGAResourceReport(luts=10000, toggle_rate=0.1)
        b = FPGAResourceReport(luts=10000, toggle_rate=0.5)
        assert b.dynamic_power_mw > a.dynamic_power_mw

    def test_total_includes_static(self):
        r = FPGAResourceReport(luts=10000, static_power_mw=100)
        assert r.total_power_mw >= 100

    def test_zero_resources_zero_dynamic(self):
        r = FPGAResourceReport(luts=0, ffs=0, bram_kb=0, dsp_slices=0)
        assert r.dynamic_power_mw == 0.0


# ── CarbonModel Tests ───────────────────────────────────────────────


class TestCarbonModel:
    def test_eu_lower_than_cn(self):
        eu = CarbonModel(GridRegion.EU)
        cn = CarbonModel(GridRegion.CN)
        assert eu.co2_g_per_kwh < cn.co2_g_per_kwh

    def test_compute_returns_grams(self):
        m = CarbonModel(GridRegion.GLOBAL)
        co2 = m.compute(power_mw=1000, duration_hours=1)
        assert co2 > 0

    def test_zero_power_zero_carbon(self):
        m = CarbonModel()
        assert m.compute(0, 100) == 0.0

    def test_annual_footprint(self):
        m = CarbonModel(GridRegion.US)
        kg = m.annual_footprint_kg(power_mw=1000)
        assert kg > 0

    def test_renewable_very_low(self):
        m = CarbonModel(GridRegion.RENEWABLE)
        assert m.co2_g_per_kwh < 50


# ── HarvestProfile Tests ────────────────────────────────────────────


class TestHarvestProfile:
    def test_default_peak_from_type(self):
        h = HarvestProfile(harvester=EnergyHarvester.SOLAR)
        assert h.peak_power_mw == 50.0

    def test_average_power(self):
        h = HarvestProfile(harvester=EnergyHarvester.PIEZO, duty_cycle=0.5)
        assert h.average_power_mw == 0.25

    def test_solar_night_zero(self):
        h = HarvestProfile(harvester=EnergyHarvester.SOLAR)
        assert h.power_at(2.0) == 0.0

    def test_solar_noon_peak(self):
        h = HarvestProfile(harvester=EnergyHarvester.SOLAR)
        noon = h.power_at(12.0)
        assert noon == pytest.approx(h.peak_power_mw)

    def test_piezo_constant(self):
        h = HarvestProfile(harvester=EnergyHarvester.PIEZO)
        assert h.power_at(0) == h.power_at(12)

    def test_energy_over_duration(self):
        h = HarvestProfile(harvester=EnergyHarvester.RF)
        energy = h.energy_over(10.0)
        assert energy == pytest.approx(h.average_power_mw * 10.0)


# ── SustainabilityOptimizer Tests ────────────────────────────────────


class TestSustainabilityOptimizer:
    def test_analyze_without_harvest(self):
        fpga = FPGAResourceReport(luts=10000, static_power_mw=50)
        opt = SustainabilityOptimizer(fpga)
        report = opt.analyze()
        assert report.deficit_mw > 0
        assert not report.net_zero_feasible
        assert any("No energy harvesting" in s for s in report.suggestions)

    def test_analyze_with_large_harvest(self):
        fpga = FPGAResourceReport(luts=10, static_power_mw=0.01)
        harvest = HarvestProfile(
            harvester=EnergyHarvester.SOLAR,
            peak_power_mw=1000,
            duty_cycle=1.0,
        )
        opt = SustainabilityOptimizer(fpga)
        report = opt.analyze(harvest)
        assert report.net_zero_feasible

    def test_duty_cycle_optimization(self):
        fpga = FPGAResourceReport(luts=50000, static_power_mw=100)
        harvest = HarvestProfile(harvester=EnergyHarvester.PIEZO)
        opt = SustainabilityOptimizer(fpga)
        report = opt.analyze(harvest)
        assert report.optimization is not None
        assert report.optimization.active_fraction <= 1.0

    def test_carbon_per_hour_positive(self):
        fpga = FPGAResourceReport(luts=10000, static_power_mw=50)
        opt = SustainabilityOptimizer(fpga)
        report = opt.analyze()
        assert report.carbon_g_per_hour >= 0

    def test_hourly_profile_length(self):
        fpga = FPGAResourceReport(luts=10000)
        harvest = HarvestProfile(harvester=EnergyHarvester.SOLAR)
        opt = SustainabilityOptimizer(fpga)
        profile = opt.hourly_profile(harvest, hours=24)
        assert len(profile) == 24
        assert all("harvest_mw" in p for p in profile)
        assert all("co2_g" in p for p in profile)

    def test_hourly_solar_night_no_harvest(self):
        fpga = FPGAResourceReport(luts=10000)
        harvest = HarvestProfile(harvester=EnergyHarvester.SOLAR)
        opt = SustainabilityOptimizer(fpga)
        profile = opt.hourly_profile(harvest, hours=24)
        assert profile[2]["harvest_mw"] == 0.0

    def test_renewable_grid_reduces_carbon(self):
        fpga = FPGAResourceReport(luts=10000, static_power_mw=50)
        opt_global = SustainabilityOptimizer(fpga, CarbonModel(GridRegion.GLOBAL))
        opt_green = SustainabilityOptimizer(fpga, CarbonModel(GridRegion.RENEWABLE))
        r1 = opt_global.analyze()
        r2 = opt_green.analyze()
        assert r2.annual_carbon_kg < r1.annual_carbon_kg


# ── Power Breakdown Tests ────────────────────────────────────────────


class TestPowerBreakdown:
    def test_breakdown_components(self):
        r = FPGAResourceReport(luts=10000, ffs=5000, bram_kb=10, dsp_slices=5)
        bd = r.power_breakdown()
        assert "lut_mw" in bd
        assert "ff_mw" in bd
        assert "bram_mw" in bd
        assert "dsp_mw" in bd
        assert "static_mw" in bd

    def test_breakdown_sums_to_total(self):
        r = FPGAResourceReport(luts=10000, ffs=5000, bram_kb=10, dsp_slices=5, static_power_mw=50)
        bd = r.power_breakdown()
        total_from_bd = sum(bd.values())
        assert abs(total_from_bd - r.total_power_mw) < 0.001

    def test_zero_resources_zero_dynamic(self):
        r = FPGAResourceReport(luts=0, ffs=0, bram_kb=0, dsp_slices=0, static_power_mw=10)
        bd = r.power_breakdown()
        assert bd["lut_mw"] == 0.0
        assert bd["static_mw"] == 10.0


# ── DVFS Tests ───────────────────────────────────────────────────────


class TestDVFS:
    def test_scale_reduces_power(self):
        r = FPGAResourceReport(luts=10000, clock_mhz=200, voltage_v=1.0)
        scaled = r.scale_dvfs(clock_mhz=100, voltage_v=0.7)
        assert scaled.total_power_mw < r.total_power_mw

    def test_scale_preserves_resources(self):
        r = FPGAResourceReport(luts=10000, ffs=5000)
        scaled = r.scale_dvfs(50, 0.6)
        assert scaled.luts == 10000
        assert scaled.ffs == 5000


# ── Vivado Parser Tests ──────────────────────────────────────────────


class TestVivadoParser:
    def test_from_vivado_dict(self):
        d = {
            "LUT": 50000,
            "FF": 30000,
            "BRAM_KB": 256,
            "DSP": 20,
            "Toggle_Rate": 0.2,
            "Clock_MHz": 150,
            "Voltage_V": 0.9,
            "Static_Power_mW": 80,
        }
        r = FPGAResourceReport.from_vivado_dict(d)
        assert r.luts == 50000
        assert r.clock_mhz == 150
        assert r.toggle_rate == 0.2

    def test_from_vivado_dict_defaults(self):
        r = FPGAResourceReport.from_vivado_dict({})
        assert r.luts == 0
        assert r.clock_mhz == 100.0


# ── EmbodiedCarbon Tests ─────────────────────────────────────────────


class TestEmbodiedCarbon:
    def test_total_embodied(self):
        ec = EmbodiedCarbon()
        assert ec.total_embodied_kg == 23.0  # 15+2+5+1

    def test_amortised_annual(self):
        ec = EmbodiedCarbon(lifetime_years=5)
        assert ec.amortised_annual_kg == pytest.approx(23.0 / 5.0)

    def test_zero_lifetime(self):
        ec = EmbodiedCarbon(lifetime_years=0)
        assert ec.amortised_annual_kg == ec.total_embodied_kg


# ── MultiHarvestStack Tests ──────────────────────────────────────────


class TestMultiHarvestStack:
    def test_add_and_count(self):
        stack = MultiHarvestStack()
        stack.add(HarvestProfile(harvester=EnergyHarvester.SOLAR))
        stack.add(HarvestProfile(harvester=EnergyHarvester.PIEZO))
        assert stack.num_sources == 2

    def test_combined_power(self):
        stack = MultiHarvestStack(
            [
                HarvestProfile(harvester=EnergyHarvester.SOLAR, peak_power_mw=50, duty_cycle=0.5),
                HarvestProfile(harvester=EnergyHarvester.PIEZO, peak_power_mw=0.5, duty_cycle=1.0),
            ]
        )
        assert stack.average_power_mw == pytest.approx(25.5)

    def test_power_at_sums(self):
        stack = MultiHarvestStack(
            [
                HarvestProfile(harvester=EnergyHarvester.PIEZO, peak_power_mw=1.0, duty_cycle=1.0),
                HarvestProfile(harvester=EnergyHarvester.RF, peak_power_mw=0.5, duty_cycle=1.0),
            ]
        )
        assert stack.power_at(12.0) == pytest.approx(1.5)

    def test_energy_over(self):
        stack = MultiHarvestStack(
            [
                HarvestProfile(harvester=EnergyHarvester.PIEZO, peak_power_mw=1.0, duty_cycle=1.0),
            ]
        )
        assert stack.energy_over(10.0) == pytest.approx(10.0)


# ── EnergyStorageSim Tests ───────────────────────────────────────────


class TestEnergyStorageSim:
    def test_initial_soc(self):
        es = EnergyStorageSim(capacity_mwh=10, initial_soc=0.5)
        assert es.soc == 0.5

    def test_charge_increases_soc(self):
        es = EnergyStorageSim(capacity_mwh=10, initial_soc=0.5, self_discharge_rate=0.0)
        es.step(net_power_mw=5.0, dt_hours=1.0)
        assert es.soc > 0.5

    def test_discharge_decreases_soc(self):
        es = EnergyStorageSim(capacity_mwh=10, initial_soc=0.5, self_discharge_rate=0.0)
        es.step(net_power_mw=-5.0, dt_hours=1.0)
        assert es.soc < 0.5

    def test_soc_clamped_at_1(self):
        es = EnergyStorageSim(capacity_mwh=1, initial_soc=0.9, self_discharge_rate=0.0)
        es.step(net_power_mw=100.0, dt_hours=1.0)
        assert es.soc <= 1.0

    def test_soc_clamped_at_0(self):
        es = EnergyStorageSim(capacity_mwh=1, initial_soc=0.1, self_discharge_rate=0.0)
        es.step(net_power_mw=-100.0, dt_hours=1.0)
        assert es.soc >= 0.0
        assert es.is_depleted

    def test_history_tracked(self):
        es = EnergyStorageSim(capacity_mwh=10, initial_soc=0.5)
        es.step(1.0)
        es.step(-1.0)
        assert len(es.history) == 3  # initial + 2 steps

    def test_energy_stored(self):
        es = EnergyStorageSim(capacity_mwh=10, initial_soc=0.8)
        assert es.energy_stored_mwh == pytest.approx(8.0)


# ── ThermalModel Tests ───────────────────────────────────────────────


class TestThermalModel:
    def test_junction_temp(self):
        tm = ThermalModel(ambient_c=25.0, r_theta_ja=15.0)
        tj = tm.junction_temp(1000)  # 1W → 25 + 15 = 40
        assert tj == pytest.approx(40.0)

    def test_is_safe(self):
        tm = ThermalModel(ambient_c=25.0, r_theta_ja=15.0, max_junction_c=85.0)
        assert tm.is_safe(1000) is True  # 40°C < 85°C
        assert tm.is_safe(5000) is False  # 25+75 = 100°C > 85°C

    def test_max_power(self):
        tm = ThermalModel(ambient_c=25.0, r_theta_ja=15.0, max_junction_c=85.0)
        mp = tm.max_power_mw()
        assert mp == pytest.approx(4000.0)


# ── Energy Efficiency Tests ──────────────────────────────────────────


class TestEnergyEfficiency:
    def test_metrics(self):
        fpga = FPGAResourceReport(luts=10000, static_power_mw=50)
        opt = SustainabilityOptimizer(fpga)
        eff = opt.energy_efficiency(ops_per_second=1e9)
        assert eff["ops_per_joule"] > 0
        assert eff["sop_per_mw"] > 0


# ── Deployment Lifetime Tests ────────────────────────────────────────


class TestDeploymentLifetime:
    def test_battery_only(self):
        fpga = FPGAResourceReport(luts=1000, static_power_mw=10)
        opt = SustainabilityOptimizer(fpga)
        lt = opt.deployment_lifetime(battery_mwh=100)
        assert lt["battery_life_hours"] > 0
        assert lt["battery_life_days"] > 0

    def test_with_harvest(self):
        fpga = FPGAResourceReport(luts=10, static_power_mw=0.01)
        harvest = HarvestProfile(
            harvester=EnergyHarvester.SOLAR, peak_power_mw=1000, duty_cycle=1.0
        )
        opt = SustainabilityOptimizer(fpga)
        lt = opt.deployment_lifetime(harvest, battery_mwh=100)
        assert lt["battery_life_hours"] == float("inf")

    def test_includes_embodied_carbon(self):
        fpga = FPGAResourceReport(luts=1000, static_power_mw=10)
        opt = SustainabilityOptimizer(fpga)
        lt = opt.deployment_lifetime()
        assert lt["annual_embodied_carbon_kg"] > 0
        assert lt["annual_total_carbon_kg"] > lt["annual_operational_carbon_kg"]


# ── Adaptive Duty Cycle Simulation Tests ─────────────────────────────


class TestAdaptiveDutyCycle:
    def test_profile_length(self):
        fpga = FPGAResourceReport(luts=10000, static_power_mw=50)
        harvest = HarvestProfile(harvester=EnergyHarvester.SOLAR)
        opt = SustainabilityOptimizer(fpga)
        timeline = opt.adaptive_duty_cycle_sim(harvest, hours=24)
        assert len(timeline) == 24

    def test_night_reduces_active(self):
        fpga = FPGAResourceReport(luts=10000, static_power_mw=50)
        harvest = HarvestProfile(harvester=EnergyHarvester.SOLAR)
        opt = SustainabilityOptimizer(fpga)
        timeline = opt.adaptive_duty_cycle_sim(harvest, hours=24, min_active=0.1)
        # At night (hour 2), solar = 0 → active_fraction = min_active
        assert timeline[2]["active_fraction"] == pytest.approx(0.1)

    def test_surplus_positive_at_noon(self):
        fpga = FPGAResourceReport(luts=100, static_power_mw=1)
        harvest = HarvestProfile(harvester=EnergyHarvester.SOLAR, peak_power_mw=100)
        opt = SustainabilityOptimizer(fpga)
        timeline = opt.adaptive_duty_cycle_sim(harvest, hours=24)
        assert timeline[12]["surplus_mw"] >= 0


# ── Storage Simulation Tests ─────────────────────────────────────────


class TestStorageSimulation:
    def test_simulate_24h(self):
        fpga = FPGAResourceReport(luts=1000, static_power_mw=1)
        harvest = HarvestProfile(harvester=EnergyHarvester.SOLAR, peak_power_mw=10)
        storage = EnergyStorageSim(capacity_mwh=10, initial_soc=0.5, self_discharge_rate=0.0)
        opt = SustainabilityOptimizer(fpga)
        timeline = opt.simulate_storage(harvest, storage, hours=24)
        assert len(timeline) == 24
        assert all("soc" in t for t in timeline)


# ── Multi-Harvest Analysis Tests ─────────────────────────────────────


class TestMultiHarvestAnalysis:
    def test_stacked_net_zero(self):
        fpga = FPGAResourceReport(luts=10, static_power_mw=0.01)
        stack = MultiHarvestStack(
            [
                HarvestProfile(harvester=EnergyHarvester.SOLAR, peak_power_mw=1000, duty_cycle=1.0),
                HarvestProfile(harvester=EnergyHarvester.PIEZO, peak_power_mw=10, duty_cycle=1.0),
            ]
        )
        report = analyze_multi_harvest(fpga, stack)
        assert report.net_zero_feasible

    def test_stacked_deficit(self):
        fpga = FPGAResourceReport(luts=100000, static_power_mw=500)
        stack = MultiHarvestStack(
            [
                HarvestProfile(harvester=EnergyHarvester.PIEZO),
                HarvestProfile(harvester=EnergyHarvester.RF),
            ]
        )
        report = analyze_multi_harvest(fpga, stack)
        assert not report.net_zero_feasible
        assert report.deficit_mw > 0


# ── Thermal Suggestion Test ──────────────────────────────────────────


class TestThermalSuggestion:
    def test_thermal_violation_suggestion(self):
        fpga = FPGAResourceReport(
            luts=100000, ffs=50000, static_power_mw=2000, clock_mhz=500, voltage_v=1.2
        )
        thermal = ThermalModel(ambient_c=40, r_theta_ja=15, max_junction_c=85)
        opt = SustainabilityOptimizer(fpga, thermal=thermal)
        report = opt.analyze()
        assert any("Thermal violation" in s for s in report.suggestions)


class TestSustainabilityEdgeBranches:
    def test_storage_step_returns_soc_when_capacity_non_positive(self):
        storage = EnergyStorageSim(capacity_mwh=0.0, initial_soc=0.5)
        assert storage.step(5.0) == storage.soc

    def test_optimise_duty_cycle_defaults_when_total_power_non_positive(self):
        opt = SustainabilityOptimizer(FPGAResourceReport(luts=500000, static_power_mw=5000))
        cfg = opt._optimize_duty_cycle(0.0, 0.0)
        assert cfg.active_fraction == 1.0

    def test_analyze_time_to_neutral_is_infinite_without_storage(self):
        opt = SustainabilityOptimizer(FPGAResourceReport(luts=500000, static_power_mw=5000))
        harvest = HarvestProfile(
            harvester=EnergyHarvester.RF, peak_power_mw=10.0, storage_capacity_mwh=0.0
        )
        report = opt.analyze(harvest=harvest)
        assert report.time_to_neutral_hours == float("inf")

    def test_deployment_lifetime_is_zero_without_battery_under_deficit(self):
        opt = SustainabilityOptimizer(FPGAResourceReport(luts=500000, static_power_mw=5000))
        harvest = HarvestProfile(harvester=EnergyHarvester.RF, peak_power_mw=10.0)
        result = opt.deployment_lifetime(harvest=harvest, battery_mwh=0.0)
        assert result["battery_life_hours"] == 0.0

    def test_adaptive_duty_cycle_runs_full_active_for_zero_power_fabric(self):
        opt = SustainabilityOptimizer(
            FPGAResourceReport(luts=0, ffs=0, bram_kb=0, dsp_slices=0, static_power_mw=0)
        )
        harvest = HarvestProfile(harvester=EnergyHarvester.RF, peak_power_mw=10.0)
        timeline = opt.adaptive_duty_cycle_sim(harvest, hours=2)
        assert all(entry["active_fraction"] == 1.0 for entry in timeline)
