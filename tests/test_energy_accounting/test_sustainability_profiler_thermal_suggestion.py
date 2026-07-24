# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalSuggestion from former test_sustainability_profiler.py

"""Focused suite: TestThermalSuggestion from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403


class TestThermalSuggestion:
    def test_thermal_violation_suggestion(self):
        fpga = FPGAResourceReport(
            luts=100000, ffs=50000, static_power_mw=2000, clock_mhz=500, voltage_v=1.2
        )
        thermal = ThermalModel(ambient_c=40, r_theta_ja=15, max_junction_c=85)
        opt = SustainabilityOptimizer(fpga, thermal=thermal)
        report = opt.analyze()
        assert any("Thermal violation" in s for s in report.suggestions)
