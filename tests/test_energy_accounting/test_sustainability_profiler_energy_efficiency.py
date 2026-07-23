# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyEfficiency from former test_sustainability_profiler.py

"""Focused suite: TestEnergyEfficiency from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403

class TestEnergyEfficiency:
    def test_metrics(self):
        fpga = FPGAResourceReport(luts=10000, static_power_mw=50)
        opt = SustainabilityOptimizer(fpga)
        eff = opt.energy_efficiency(ops_per_second=1e9)
        assert eff["ops_per_joule"] > 0
        assert eff["sop_per_mw"] > 0
