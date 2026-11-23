# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyStorageSim from former test_sustainability_profiler.py

"""Focused suite: TestEnergyStorageSim from former test_sustainability_profiler.py."""

from __future__ import annotations

import pytest
from sc_neurocore.energy_accounting.sustainability_profiler import (
    EnergyStorageSim,
)


class TestEnergyStorageSim:
    """State of charge of simulated energy storage."""

    def test_initial_soc(self) -> None:
        """Storage starts half charged."""
        es = EnergyStorageSim(capacity_mwh=10, initial_soc=0.5)
        assert es.soc == 0.5

    def test_charge_increases_soc(self) -> None:
        """Charging raises the state of charge."""
        es = EnergyStorageSim(capacity_mwh=10, initial_soc=0.5, self_discharge_rate=0.0)
        es.step(net_power_mw=5.0, dt_hours=1.0)
        assert es.soc > 0.5

    def test_discharge_decreases_soc(self) -> None:
        """Discharging lowers the state of charge."""
        es = EnergyStorageSim(capacity_mwh=10, initial_soc=0.5, self_discharge_rate=0.0)
        es.step(net_power_mw=-5.0, dt_hours=1.0)
        assert es.soc < 0.5

    def test_soc_clamped_at_1(self) -> None:
        """The state of charge never exceeds one."""
        es = EnergyStorageSim(capacity_mwh=1, initial_soc=0.9, self_discharge_rate=0.0)
        es.step(net_power_mw=100.0, dt_hours=1.0)
        assert es.soc <= 1.0

    def test_soc_clamped_at_0(self) -> None:
        """The state of charge never drops below zero and then is depleted."""
        es = EnergyStorageSim(capacity_mwh=1, initial_soc=0.1, self_discharge_rate=0.0)
        es.step(net_power_mw=-100.0, dt_hours=1.0)
        assert es.soc >= 0.0
        assert es.is_depleted

    def test_history_tracked(self) -> None:
        """The history holds the start and each step."""
        es = EnergyStorageSim(capacity_mwh=10, initial_soc=0.5)
        es.step(1.0)
        es.step(-1.0)
        assert len(es.history) == 3  # initial + 2 steps

    def test_energy_stored(self) -> None:
        """Stored energy follows capacity and state of charge."""
        es = EnergyStorageSim(capacity_mwh=10, initial_soc=0.8)
        assert es.energy_stored_mwh == pytest.approx(8.0)
