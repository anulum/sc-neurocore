# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpintronicVerilogGenerator from former test_spintronic_mapper.py

"""Focused suite: TestSpintronicVerilogGenerator from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403

class TestSpintronicVerilogGenerator:
    def test_generate(self):
        v = SpintronicVerilogGenerator.generate(
            "sc_spin_array",
            8,
            16,
            SpintronicTech.SOT_MRAM,
        )
        assert "module sc_spin_array" in v
        assert "ROWS = 8" in v
        assert "COLS = 16" in v

    def test_has_programming_interface(self):
        v = SpintronicVerilogGenerator.generate(
            "test",
            4,
            4,
            SpintronicTech.SKYRMION,
        )
        assert "prog_en" in v
        assert "prog_weight" in v
