# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGeneratePropertySby from former test_formal_property_check.py

"""Focused suite: TestGeneratePropertySby from former test_formal_property_check.py."""

from __future__ import annotations

from tests.formal_property_check_support import *  # noqa: F403

class TestGeneratePropertySby:
    """The generated ``.sby`` script (no run required)."""

    def test_reads_rtl_then_sva_and_preps_top(self) -> None:
        sby = formal_property_check._generate_property_sby(
            "mon", "mon.v", "mon_sva.sv", depth=18, mode="bmc", engine="z3"
        )
        assert "mode bmc" in sby
        assert "bmc: depth 18" in sby
        assert "smtbmc z3" in sby
        assert sby.index("read -formal mon.v") < sby.index("read -sv -formal mon_sva.sv")
        assert "prep -top mon" in sby
