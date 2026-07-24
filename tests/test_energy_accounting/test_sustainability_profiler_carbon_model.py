# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCarbonModel from former test_sustainability_profiler.py

"""Focused suite: TestCarbonModel from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403


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
