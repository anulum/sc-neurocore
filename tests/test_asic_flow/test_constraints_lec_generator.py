# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLECGenerator from former test_constraints.py

"""Focused suite: TestLECGenerator from former test_constraints.py."""

from __future__ import annotations

from tests.test_asic_flow.constraints_support import *  # noqa: F403

class TestLECGenerator:
    def test_generates_lec(self) -> None:
        design = DesignParams(top_module="sc_lif")
        script = LECGenerator.generate(design)
        assert "equiv_make" in script
        assert "equiv_status" in script
        assert "sc_lif" in script
