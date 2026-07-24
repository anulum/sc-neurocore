# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIRDropGenerator from former test_constraints.py

"""Focused suite: TestIRDropGenerator from former test_constraints.py."""

from __future__ import annotations

from tests.test_asic_flow.constraints_support import *  # noqa: F403


class TestIRDropGenerator:
    def test_generates_script(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = IRDropGenerator.generate(pdk, design, toggle_rate=0.15)
        assert "analyze_power_grid" in script
        assert "0.150" in script
        assert "VDD" in script
