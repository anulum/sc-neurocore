# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSustainabilityEdgeBranches from former test_sustainability_profiler.py

"""Focused suite: TestSustainabilityEdgeBranches from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403

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
