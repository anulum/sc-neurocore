# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSustainabilityOptimizer from former test_sustainability_profiler.py

"""Focused suite: TestSustainabilityOptimizer from former test_sustainability_profiler.py."""

from __future__ import annotations

from sc_neurocore.energy_accounting.sustainability_profiler import (
    CarbonModel,
    EnergyHarvester,
    FPGAResourceReport,
    GridRegion,
    HarvestProfile,
    SustainabilityOptimizer,
)


class TestSustainabilityOptimizer:
    """Net-zero analysis and duty-cycle optimisation."""

    def test_analyze_without_harvest(self) -> None:
        """Without harvest there is a deficit and a harvesting suggestion."""
        fpga = FPGAResourceReport(luts=10000, static_power_mw=50)
        opt = SustainabilityOptimizer(fpga)
        report = opt.analyze()
        assert report.deficit_mw > 0
        assert not report.net_zero_feasible
        assert any("No energy harvesting" in s for s in report.suggestions)

    def test_analyze_with_large_harvest(self) -> None:
        """A large harvest reaches net zero."""
        fpga = FPGAResourceReport(luts=10, static_power_mw=0.01)
        harvest = HarvestProfile(
            harvester=EnergyHarvester.SOLAR,
            peak_power_mw=1000,
            duty_cycle=1.0,
        )
        opt = SustainabilityOptimizer(fpga)
        report = opt.analyze(harvest)
        assert report.net_zero_feasible

    def test_duty_cycle_optimization(self) -> None:
        """Optimisation proposes an active fraction of at most one."""
        fpga = FPGAResourceReport(luts=50000, static_power_mw=100)
        harvest = HarvestProfile(harvester=EnergyHarvester.PIEZO)
        opt = SustainabilityOptimizer(fpga)
        report = opt.analyze(harvest)
        assert report.optimization is not None
        assert report.optimization.active_fraction <= 1.0

    def test_carbon_per_hour_positive(self) -> None:
        """Hourly carbon is not negative."""
        fpga = FPGAResourceReport(luts=10000, static_power_mw=50)
        opt = SustainabilityOptimizer(fpga)
        report = opt.analyze()
        assert report.carbon_g_per_hour >= 0

    def test_hourly_profile_length(self) -> None:
        """The hourly profile covers 24 hours with harvest and carbon."""
        fpga = FPGAResourceReport(luts=10000)
        harvest = HarvestProfile(harvester=EnergyHarvester.SOLAR)
        opt = SustainabilityOptimizer(fpga)
        profile = opt.hourly_profile(harvest, hours=24)
        assert len(profile) == 24
        assert all("harvest_mw" in p for p in profile)
        assert all("co2_g" in p for p in profile)

    def test_hourly_solar_night_no_harvest(self) -> None:
        """Solar harvests nothing at night in the profile."""
        fpga = FPGAResourceReport(luts=10000)
        harvest = HarvestProfile(harvester=EnergyHarvester.SOLAR)
        opt = SustainabilityOptimizer(fpga)
        profile = opt.hourly_profile(harvest, hours=24)
        assert profile[2]["harvest_mw"] == 0.0

    def test_renewable_grid_reduces_carbon(self) -> None:
        """A renewable grid lowers the annual carbon."""
        fpga = FPGAResourceReport(luts=10000, static_power_mw=50)
        opt_global = SustainabilityOptimizer(fpga, CarbonModel(GridRegion.GLOBAL))
        opt_green = SustainabilityOptimizer(fpga, CarbonModel(GridRegion.RENEWABLE))
        r1 = opt_global.analyze()
        r2 = opt_green.analyze()
        assert r2.annual_carbon_kg < r1.annual_carbon_kg
