# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMemoryFootprint from former test_edge.py

"""Focused suite: TestMemoryFootprint from former test_edge.py."""

from __future__ import annotations

from edge_support import *  # noqa: F403

class TestMemoryFootprint:
    def test_fits_in_ram(self):
        fp = MemoryFootprint.estimate(2, 16, 8, Board.ESP32_C6)
        assert fp.fits_in_ram
        assert fp.fits_in_flash
        assert fp.total_bytes > 0

    def test_max_neurons(self):
        n_esp = MemoryFootprint.max_neurons(Board.ESP32_C6)
        n_gd = MemoryFootprint.max_neurons(Board.GD32VF103)
        assert n_esp > n_gd > 0

    def test_max_neurons_returns_zero_when_ram_not_above_overhead(self):
        class TinyBoard:
            ram_kb = 0

        assert MemoryFootprint.max_neurons(TinyBoard()) == 0
