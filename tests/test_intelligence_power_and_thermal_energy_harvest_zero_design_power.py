# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyHarvestZeroDesignPower from former test_intelligence_power_and_thermal.py

"""Focused suite: TestEnergyHarvestZeroDesignPower from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from sc_neurocore.compiler.intelligence import (
    model_energy_harvest,
)


class TestEnergyHarvestZeroDesignPower:
    """A non-positive design power has no ratio.

    The harvester budget reports a full duty cycle and saturated margin
    rather than dividing by zero.
    """

    def test_zero_design_power_assumes_full_duty(self) -> None:
        """Zero design power assumes a full duty cycle."""
        budget = model_energy_harvest(0.0, harvester_type="solar", environment="outdoor")
        assert budget.recommended_duty_cycle == 1.0
        assert budget.margin_pct == 100.0
        assert budget.energy_positive is True
