# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIRDropModel from former test_memristor_mapper.py

"""Focused suite: TestIRDropModel from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403

class TestIRDropModel:
    def test_corner_no_drop(self) -> None:
        ir = IRDropModel()
        assert ir.voltage_drop(0, 0) == 0.0

    def test_drop_increases_with_position(self) -> None:
        ir = IRDropModel()
        d1 = ir.voltage_drop(10, 10)
        d2 = ir.voltage_drop(50, 50)
        assert d2 > d1

    def test_effective_conductance_reduced(self) -> None:
        ir = IRDropModel(r_wire_per_cell=5.0)
        g_nom = 50e-6
        g_eff = ir.effective_conductance(g_nom, 100, 100, v_read=0.2)
        assert g_eff < g_nom

    def test_zero_drop_at_corner(self) -> None:
        ir = IRDropModel()
        g_nom = 50e-6
        g_eff = ir.effective_conductance(g_nom, 0, 0)
        assert g_eff == g_nom
