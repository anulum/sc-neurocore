# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIOConstraints from former test_constraints.py

"""Focused suite: TestIOConstraints from former test_constraints.py."""

from __future__ import annotations

from tests.test_asic_flow.constraints_support import *  # noqa: F403

class TestIOConstraints:
    def test_generate(self) -> None:
        pins = [IOPin("clk", "input", "N"), IOPin("data_out", "output", "S")]
        design = DesignParams()
        script = IOConstraintGenerator.generate(pins, design)
        assert "clk" in script
        assert "data_out" in script

    def test_auto_assign(self) -> None:
        names = ["a", "b", "c", "d", "e"]
        pins = IOConstraintGenerator.auto_assign(names)
        assert len(pins) == 5
        sides = set(p.side for p in pins)
        assert len(sides) == 4
