# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCarbonModel from former test_sustainability_profiler.py

"""Focused suite: TestCarbonModel from former test_sustainability_profiler.py."""

from __future__ import annotations

from sc_neurocore.energy_accounting.sustainability_profiler import (
    CarbonModel,
    GridRegion,
)


class TestCarbonModel:
    """Grid carbon intensity and emissions."""

    def test_eu_lower_than_cn(self) -> None:
        """The EU grid emits less carbon per kilowatt-hour than China's."""
        eu = CarbonModel(GridRegion.EU)
        cn = CarbonModel(GridRegion.CN)
        assert eu.co2_g_per_kwh < cn.co2_g_per_kwh

    def test_compute_returns_grams(self) -> None:
        """Running at power for a time emits carbon."""
        m = CarbonModel(GridRegion.GLOBAL)
        co2 = m.compute(power_mw=1000, duration_hours=1)
        assert co2 > 0

    def test_zero_power_zero_carbon(self) -> None:
        """Zero power emits no carbon."""
        m = CarbonModel()
        assert m.compute(0, 100) == 0.0

    def test_annual_footprint(self) -> None:
        """The annual footprint is positive."""
        m = CarbonModel(GridRegion.US)
        kg = m.annual_footprint_kg(power_mw=1000)
        assert kg > 0

    def test_renewable_very_low(self) -> None:
        """A renewable grid emits under 50 g per kilowatt-hour."""
        m = CarbonModel(GridRegion.RENEWABLE)
        assert m.co2_g_per_kwh < 50
