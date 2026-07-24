# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSkyrmionHall from former test_spintronic_mapper.py

"""Focused suite: TestSkyrmionHall from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403


class TestSkyrmionHall:
    def test_hall_angle(self):
        shc = SkyrmionHallCorrector()
        assert shc.hall_angle_deg > 0

    def test_corrected_position(self):
        shc = SkyrmionHallCorrector()
        x, y = shc.corrected_position(100.0, 50.0)
        assert x == 100.0
        assert abs(y) <= 25.0  # clamped to track width

    def test_needs_confinement(self):
        shc = SkyrmionHallCorrector()
        assert isinstance(shc.needs_confinement, bool)
