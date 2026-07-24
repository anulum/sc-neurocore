# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFloorplanner from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestFloorplanner from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations

from tests.intelligence_soc_and_chiplet_support import *  # noqa: F403


class TestFloorplanner:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import plan_multi_die_floorplan

        r = plan_multi_die_floorplan(
            {"cortex_a": 500, "cortex_b": 300, "cortex_c": 400},
            die_capacity=1000,
        )
        assert len(r.die_assignment) == 3
        assert r.total_dies >= 1

    def test_overflow(self):
        from sc_neurocore.compiler.intelligence import plan_multi_die_floorplan

        r = plan_multi_die_floorplan(
            {"big": 900, "huge": 800},
            die_capacity=1000,
            num_dies=2,
        )
        assert r.total_dies == 2
