# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiHarvestStack from former test_sustainability_profiler.py

"""Focused suite: TestMultiHarvestStack from former test_sustainability_profiler.py."""

from __future__ import annotations

import pytest
from sc_neurocore.energy_accounting.sustainability_profiler import (
    EnergyHarvester,
    HarvestProfile,
    MultiHarvestStack,
)


class TestMultiHarvestStack:
    """Stacks of several energy harvesters."""

    def test_add_and_count(self) -> None:
        """Added sources are counted."""
        stack = MultiHarvestStack()
        stack.add(HarvestProfile(harvester=EnergyHarvester.SOLAR))
        stack.add(HarvestProfile(harvester=EnergyHarvester.PIEZO))
        assert stack.num_sources == 2

    def test_combined_power(self) -> None:
        """The stack's average power sums its sources."""
        stack = MultiHarvestStack(
            [
                HarvestProfile(harvester=EnergyHarvester.SOLAR, peak_power_mw=50, duty_cycle=0.5),
                HarvestProfile(harvester=EnergyHarvester.PIEZO, peak_power_mw=0.5, duty_cycle=1.0),
            ]
        )
        assert stack.average_power_mw == pytest.approx(25.5)

    def test_power_at_sums(self) -> None:
        """The stack's power at an hour sums its sources."""
        stack = MultiHarvestStack(
            [
                HarvestProfile(harvester=EnergyHarvester.PIEZO, peak_power_mw=1.0, duty_cycle=1.0),
                HarvestProfile(harvester=EnergyHarvester.RF, peak_power_mw=0.5, duty_cycle=1.0),
            ]
        )
        assert stack.power_at(12.0) == pytest.approx(1.5)

    def test_energy_over(self) -> None:
        """The stack's energy over a duration sums its sources."""
        stack = MultiHarvestStack(
            [
                HarvestProfile(harvester=EnergyHarvester.PIEZO, peak_power_mw=1.0, duty_cycle=1.0),
            ]
        )
        assert stack.energy_over(10.0) == pytest.approx(10.0)
