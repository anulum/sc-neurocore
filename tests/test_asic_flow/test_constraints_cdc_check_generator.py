# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCDCCheckGenerator from former test_constraints.py

"""Focused suite: TestCDCCheckGenerator from former test_constraints.py."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import (
    CDCCheckGenerator,
    DesignParams,
)


class TestCDCCheckGenerator:
    """Clock-domain-crossing check scripts."""

    def test_single_domain(self) -> None:
        """A single clock domain still gets a CDC report."""
        design = DesignParams(clock_name="clk")
        script = CDCCheckGenerator.generate(design)
        assert "clk" in script
        assert "report_cdc" in script

    def test_multi_domain(self) -> None:
        """Both clock domains appear in the CDC script."""
        design = DesignParams()
        script = CDCCheckGenerator.generate(design, clock_domains=["clk_fast", "clk_slow"])
        assert "clk_fast" in script
        assert "clk_slow" in script
