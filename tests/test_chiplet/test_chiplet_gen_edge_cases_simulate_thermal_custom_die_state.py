# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSimulateThermalCustomDieState from former test_chiplet_gen_edge_cases.py

"""Focused suite: TestSimulateThermalCustomDieState from former test_chiplet_gen_edge_cases.py."""

from __future__ import annotations

from chiplet_gen_edge_cases_support import *  # noqa: F403


class TestSimulateThermalCustomDieState:
    def test_uses_provided_die_state_dict(self) -> None:
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0))
        topo.add_die(ChipletDie(die_id=1))
        # Override only die 0; die 1 falls through to default branch.
        custom = {0: DieThermal(die_id=0, r_to_ambient_k_per_w=10.0)}
        report = simulate_thermal(
            topo,
            power_per_die_mw={0: 1000.0, 1: 100.0},
            die_state=custom,
        )
        assert report is not None
        # Die 0's higher R_thermal → higher temperature than die 1.
        t0 = report.die_temps[0]
        t1 = report.die_temps[1]
        assert t0 > t1
