# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestClockDomainCrossingZero from former test_chiplet_gen_edge_cases.py

"""Focused suite: TestClockDomainCrossingZero from former test_chiplet_gen_edge_cases.py."""

from __future__ import annotations

from sc_neurocore.chiplet.chiplet_gen import (
    CDCConfig,
)


class TestClockDomainCrossingZero:
    """The clock-domain ratio when the destination clock is zero."""

    def test_ratio_returns_one_when_dst_clock_zero(self) -> None:
        """A zero destination clock gives a ratio of one instead of dividing by zero."""
        cdc = CDCConfig(src_clk_mhz=200.0, dst_clk_mhz=0.0)
        assert cdc.ratio == 1.0

    def test_ratio_uses_real_division_when_dst_nonzero(self) -> None:
        """A non-zero destination clock gives the true clock ratio."""
        cdc = CDCConfig(src_clk_mhz=200.0, dst_clk_mhz=100.0)
        assert cdc.ratio == 2.0
