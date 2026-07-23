# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestClockDomainCrossingZero from former test_chiplet_gen_edge_cases.py

"""Focused suite: TestClockDomainCrossingZero from former test_chiplet_gen_edge_cases.py."""

from __future__ import annotations

from chiplet_gen_edge_cases_support import *  # noqa: F403

class TestClockDomainCrossingZero:
    def test_ratio_returns_one_when_dst_clock_zero(self) -> None:
        cdc = CDCConfig(src_clk_mhz=200.0, dst_clk_mhz=0.0)
        assert cdc.ratio == 1.0

    def test_ratio_uses_real_division_when_dst_nonzero(self) -> None:
        cdc = CDCConfig(src_clk_mhz=200.0, dst_clk_mhz=100.0)
        assert cdc.ratio == 2.0
