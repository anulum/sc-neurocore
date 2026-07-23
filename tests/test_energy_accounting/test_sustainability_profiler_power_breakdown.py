# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerBreakdown from former test_sustainability_profiler.py

"""Focused suite: TestPowerBreakdown from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403

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
