# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalPhaseShifter from former test_photonic_noc.py

"""Focused suite: TestThermalPhaseShifter from former test_photonic_noc.py."""

from __future__ import annotations

from photonic_noc_support import *  # noqa: F403


class TestThermalPhaseShifter:
    """Thermal tuning model tests."""

    def test_power_for_pi(self) -> None:
        tps = ThermalPhaseShifter()
        p = tps.power_for_phase(math.pi)
        assert p > 0

    def test_power_scales_with_phase(self) -> None:
        tps = ThermalPhaseShifter()
        p1 = tps.power_for_phase(math.pi / 4)
        p2 = tps.power_for_phase(math.pi / 2)
        assert p2 > p1

    def test_analyze_design(self, simple_design: PhotonicCircuitDesign) -> None:
        tps = ThermalPhaseShifter()
        result = tps.analyze_design(simple_design)
        assert result["total_power_mw"] > 0
        assert result["n_gates"] == len(simple_design.mzi_gates)
