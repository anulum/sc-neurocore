# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSEUScrubber from former test_intelligence_digital_twin.py

"""Focused suite: TestSEUScrubber from former test_intelligence_digital_twin.py."""

from __future__ import annotations

from tests.intelligence_digital_twin_support import *  # noqa: F403


class TestSEUScrubber:
    def test_leo(self):
        from sc_neurocore.compiler.intelligence import schedule_seu_scrubbing

        s = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=400)
        assert s.interval_ms > 0
        assert s.frames_per_cycle > 0
        assert s.strategy == "hybrid"

    def test_higher_orbit(self):
        from sc_neurocore.compiler.intelligence import schedule_seu_scrubbing

        leo = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=400)
        geo = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=35786)
        # Higher orbit = more flux = shorter interval
        assert geo.interval_ms < leo.interval_ms
