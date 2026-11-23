# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerProfile from former test_edge.py

"""Focused suite: TestPowerProfile from former test_edge.py."""

from __future__ import annotations

from sc_neurocore.edge import (
    Board,
)
from sc_neurocore.edge.power_estimator import (
    PowerProfile,
)


class TestPowerProfile:
    """Power profiles of edge boards."""

    def test_creation(self) -> None:
        """A profile keeps its active and sleep power."""
        pp = PowerProfile.for_board(Board.ESP32_C6, 160)
        assert pp.active_uw == 18_000
        assert pp.sleep_uw == 7

    def test_scaled_with_clock(self) -> None:
        """A lower clock draws less active power."""
        pp80 = PowerProfile.for_board(Board.ESP32_C6, 80)
        pp160 = PowerProfile.for_board(Board.ESP32_C6, 160)
        assert pp80.active_uw < pp160.active_uw

    def test_duty_cycled(self) -> None:
        """Power falls from full activity through half duty to sleep."""
        pp = PowerProfile.for_board(Board.ESP32_C6, 160)
        full = pp.duty_cycled_uw(1.0)
        half = pp.duty_cycled_uw(0.5)
        sleep = pp.duty_cycled_uw(0.0)
        assert full > half > sleep

    def test_all_boards(self) -> None:
        """Every board has positive active and sleep power."""
        for board in Board:
            pp = PowerProfile.for_board(board, 160)
            assert pp.active_uw > 0
            assert pp.sleep_uw > 0
