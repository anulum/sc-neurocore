# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStorageSimulation from former test_sustainability_profiler.py

"""Focused suite: TestStorageSimulation from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403


class TestStorageSimulation:
    def test_simulate_24h(self):
        fpga = FPGAResourceReport(luts=1000, static_power_mw=1)
        harvest = HarvestProfile(harvester=EnergyHarvester.SOLAR, peak_power_mw=10)
        storage = EnergyStorageSim(capacity_mwh=10, initial_soc=0.5, self_discharge_rate=0.0)
        opt = SustainabilityOptimizer(fpga)
        timeline = opt.simulate_storage(harvest, storage, hours=24)
        assert len(timeline) == 24
        assert all("soc" in t for t in timeline)
