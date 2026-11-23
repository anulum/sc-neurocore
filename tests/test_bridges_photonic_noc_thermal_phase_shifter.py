# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalPhaseShifter from former test_bridges_photonic_noc.py

"""Focused suite: TestThermalPhaseShifter from former test_bridges_photonic_noc.py."""

from __future__ import annotations

import math
import pytest
from sc_neurocore.bridges.photonic_noc import (
    ThermalPhaseShifter,
)


class TestThermalPhaseShifter:
    """Heater power of a thermal phase shifter."""

    def test_power_for_pi_phase_positive(self) -> None:
        """A pi shift needs positive power."""
        shifter = ThermalPhaseShifter()
        power = shifter.power_for_phase(math.pi)
        assert power > 0

    def test_power_for_zero_phase(self) -> None:
        """No shift needs no power."""
        shifter = ThermalPhaseShifter()
        power = shifter.power_for_phase(0.0)
        assert power == pytest.approx(0.0, abs=1e-6)
