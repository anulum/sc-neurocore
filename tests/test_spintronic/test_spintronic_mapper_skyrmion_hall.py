# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSkyrmionHall from former test_spintronic_mapper.py

"""Focused suite: TestSkyrmionHall from former test_spintronic_mapper.py."""

from __future__ import annotations

from sc_neurocore.spintronic.spintronic_mapper import (
    SkyrmionHallCorrector,
)


class TestSkyrmionHall:
    """Correcting the skyrmion Hall drift."""

    def test_hall_angle(self) -> None:
        """The skyrmion Hall angle is positive."""
        shc = SkyrmionHallCorrector()
        assert shc.hall_angle_deg > 0

    def test_corrected_position(self) -> None:
        """Correction keeps x and clamps y to the track width."""
        shc = SkyrmionHallCorrector()
        x, y = shc.corrected_position(100.0, 50.0)
        assert x == 100.0
        assert abs(y) <= 25.0  # clamped to track width

    def test_needs_confinement(self) -> None:
        """The confinement flag is a boolean; its value is not checked."""
        shc = SkyrmionHallCorrector()
        assert isinstance(shc.needs_confinement, bool)
