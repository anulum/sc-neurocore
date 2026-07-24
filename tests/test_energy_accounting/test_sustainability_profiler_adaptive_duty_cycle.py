# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveDutyCycle from former test_sustainability_profiler.py

"""Focused suite: TestAdaptiveDutyCycle from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403


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
