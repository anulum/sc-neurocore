# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalConductanceMissingDie from former test_chiplet_gen_edge_cases.py

"""Focused suite: TestThermalConductanceMissingDie from former test_chiplet_gen_edge_cases.py."""

from __future__ import annotations

from chiplet_gen_edge_cases_support import *  # noqa: F403

class TestThermalConductanceMissingDie:
    def test_simulate_thermal_skips_link_with_missing_die(self) -> None:
        # Dies 0 and 1; link references die 99.
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0))
        topo.add_die(ChipletDie(die_id=1))
        topo.add_link(InterposerLink.from_tech(0, 99, InterposerTech.UCIE))
        # Should run without raising despite the dangling link.
        report = simulate_thermal(topo, power_per_die_mw={0: 100.0, 1: 100.0})
        assert report is not None
