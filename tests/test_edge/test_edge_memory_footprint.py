# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMemoryFootprint from former test_edge.py

"""Focused suite: TestMemoryFootprint from former test_edge.py."""

from __future__ import annotations

from typing import cast

from sc_neurocore.edge import (
    Board,
)
from sc_neurocore.edge.power_estimator import (
    MemoryFootprint,
)


class TestMemoryFootprint:
    """Memory footprint of networks on edge boards."""

    def test_fits_in_ram(self) -> None:
        """A small network fits in the board's RAM and flash."""
        fp = MemoryFootprint.estimate(2, 16, 8, Board.ESP32_C6)
        assert fp.fits_in_ram
        assert fp.fits_in_flash
        assert fp.total_bytes > 0

    def test_max_neurons(self) -> None:
        """A larger board holds more neurons than a smaller one."""
        n_esp = MemoryFootprint.max_neurons(Board.ESP32_C6)
        n_gd = MemoryFootprint.max_neurons(Board.GD32VF103)
        assert n_esp > n_gd > 0

    def test_max_neurons_returns_zero_when_ram_not_above_overhead(self) -> None:
        """A board without RAM beyond the overhead holds no neurons."""

        class TinyBoard:
            ram_kb = 0

        assert MemoryFootprint.max_neurons(cast(Board, TinyBoard())) == 0
