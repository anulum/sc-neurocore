# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCdcConfigsMissingDie from former test_chiplet_gen_edge_cases.py

"""Focused suite: TestCdcConfigsMissingDie from former test_chiplet_gen_edge_cases.py."""

from __future__ import annotations

from chiplet_gen_edge_cases_support import *  # noqa: F403


class TestCdcConfigsMissingDie:
    def test_continues_when_link_references_unknown_die(self) -> None:
        # Topology has dies 0, 1; link references die 99 (missing).
        topo = ChipletTopology()
        topo.add_die(ChipletDie(die_id=0, clock_mhz=100.0))
        topo.add_die(ChipletDie(die_id=1, clock_mhz=100.0))
        topo.add_link(InterposerLink.from_tech(0, 99, InterposerTech.UCIE))
        topo.add_link(InterposerLink.from_tech(0, 1, InterposerTech.UCIE))
        cfgs = compute_cdc_configs(topo)
        # Only the (0, 1) link should produce a CDCConfig.
        assert (0, 1) in cfgs
        assert (0, 99) not in cfgs
