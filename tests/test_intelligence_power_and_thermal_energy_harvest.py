# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyHarvest from former test_intelligence_power_and_thermal.py

"""Focused suite: TestEnergyHarvest from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from tests.intelligence_power_and_thermal_support import *  # noqa: F403


class TestEnergyHarvest(unittest.TestCase):
    def test_solar_outdoor(self):
        r = model_energy_harvest(
            100.0, harvester_type="solar", environment="outdoor", harvester_area_cm2=1.0
        )
        self.assertTrue(r.energy_positive)
        self.assertGreater(r.margin_pct, 0)

    def test_rf_indoor_insufficient(self):
        r = model_energy_harvest(100.0, harvester_type="rf", environment="indoor")
        self.assertFalse(r.energy_positive)
        self.assertLess(r.recommended_duty_cycle, 1.0)
