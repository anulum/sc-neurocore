# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeploymentLifetime from former test_sustainability_profiler.py

"""Focused suite: TestDeploymentLifetime from former test_sustainability_profiler.py."""

from __future__ import annotations

from sc_neurocore.energy_accounting.sustainability_profiler import (
    EnergyHarvester,
    FPGAResourceReport,
    HarvestProfile,
    SustainabilityOptimizer,
)


class TestDeploymentLifetime:
    """Battery life and carbon over a deployment."""

    def test_battery_only(self) -> None:
        """A battery alone lasts a positive number of hours and days."""
        fpga = FPGAResourceReport(luts=1000, static_power_mw=10)
        opt = SustainabilityOptimizer(fpga)
        lt = opt.deployment_lifetime(battery_mwh=100)
        assert lt["battery_life_hours"] > 0
        assert lt["battery_life_days"] > 0

    def test_with_harvest(self) -> None:
        """Enough harvest makes the battery last indefinitely."""
        fpga = FPGAResourceReport(luts=10, static_power_mw=0.01)
        harvest = HarvestProfile(
            harvester=EnergyHarvester.SOLAR, peak_power_mw=1000, duty_cycle=1.0
        )
        opt = SustainabilityOptimizer(fpga)
        lt = opt.deployment_lifetime(harvest, battery_mwh=100)
        assert lt["battery_life_hours"] == float("inf")

    def test_includes_embodied_carbon(self) -> None:
        """Annual carbon includes the embodied share."""
        fpga = FPGAResourceReport(luts=1000, static_power_mw=10)
        opt = SustainabilityOptimizer(fpga)
        lt = opt.deployment_lifetime()
        assert lt["annual_embodied_carbon_kg"] > 0
        assert lt["annual_total_carbon_kg"] > lt["annual_operational_carbon_kg"]
