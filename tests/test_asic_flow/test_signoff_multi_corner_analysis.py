# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiCornerAnalysis from former test_signoff.py

"""Focused suite: TestMultiCornerAnalysis from former test_signoff.py."""

from __future__ import annotations

from tests.test_asic_flow.signoff_support import *  # noqa: F403

class TestMultiCornerAnalysis:
    def test_generates_all_corners(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = MultiCornerAnalysis.generate(pdk, design)
        assert "ss_125C" in script
        assert "ff_" in script

    def test_custom_corners(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        corners = [PVTCorner(CornerType.TT, 25.0, 1.8, "_tt_025C_1v80")]
        script = MultiCornerAnalysis.generate(pdk, design, corners)
        assert "tt_25C" in script

    def test_worst_slack(self) -> None:
        wns = {"tt": 0.5, "ss": -0.2, "ff": 1.0}
        corner, slack = MultiCornerAnalysis.worst_slack(wns)
        assert corner == "ss"
        assert slack == -0.2

    def test_empty_slack_map_returns_neutral_sentinel(self) -> None:
        """No reported corners produce the documented neutral sentinel."""
        assert MultiCornerAnalysis.worst_slack({}) == ("none", 0.0)

    def test_corner_without_suffix_uses_configured_liberty(self) -> None:
        """A custom corner without a suffix leaves the Liberty path unchanged."""
        pdk = PDKConfig.from_pdk_type(PDKType.CUSTOM)
        pdk.liberty_file = "custom.lib"
        corner = PVTCorner(CornerType.TT, 25.0, 1.8)

        script = MultiCornerAnalysis.generate(pdk, DesignParams(), [corner])

        assert "read_liberty custom.lib" in script

    def test_pvt_corner_label(self) -> None:
        c = PVTCorner(CornerType.SS, 125.0, 1.62)
        assert "ss" in c.label
        assert "125" in c.label
