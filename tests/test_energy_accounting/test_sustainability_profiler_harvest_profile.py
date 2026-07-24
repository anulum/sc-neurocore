# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHarvestProfile from former test_sustainability_profiler.py

"""Focused suite: TestHarvestProfile from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403


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
