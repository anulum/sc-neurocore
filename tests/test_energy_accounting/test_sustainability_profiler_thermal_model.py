# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalModel from former test_sustainability_profiler.py

"""Focused suite: TestThermalModel from former test_sustainability_profiler.py."""

from __future__ import annotations

import pytest
from sc_neurocore.energy_accounting.sustainability_profiler import (
    ThermalModel,
)


class TestThermalModel:
    """Junction temperature and thermal limits."""

    def test_junction_temp(self) -> None:
        """Power and thermal resistance give the junction temperature."""
        tm = ThermalModel(ambient_c=25.0, r_theta_ja=15.0)
        tj = tm.junction_temp(1000)  # 1W → 25 + 15 = 40
        assert tj == pytest.approx(40.0)

    def test_is_safe(self) -> None:
        """Power is safe below the junction limit and unsafe above it."""
        tm = ThermalModel(ambient_c=25.0, r_theta_ja=15.0, max_junction_c=85.0)
        assert tm.is_safe(1000) is True  # 40°C < 85°C
        assert tm.is_safe(5000) is False  # 25+75 = 100°C > 85°C

    def test_max_power(self) -> None:
        """The model reports the maximum safe power."""
        tm = ThermalModel(ambient_c=25.0, r_theta_ja=15.0, max_junction_c=85.0)
        mp = tm.max_power_mw()
        assert mp == pytest.approx(4000.0)
