# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiHarvestAnalysis from former test_sustainability_profiler.py

"""Focused suite: TestMultiHarvestAnalysis from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403

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
