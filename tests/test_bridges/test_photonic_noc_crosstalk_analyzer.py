# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrosstalkAnalyzer from former test_photonic_noc.py

"""Focused suite: TestCrosstalkAnalyzer from former test_photonic_noc.py."""

from __future__ import annotations

from sc_neurocore.bridges.photonic_noc import (
    CrosstalkAnalyzer,
    PhotonicCircuitDesign,
)


class TestCrosstalkAnalyzer:
    """WDM crosstalk tests."""

    def test_analyze(self, simple_design: PhotonicCircuitDesign) -> None:
        """Four channels give a negative worst crosstalk."""
        ct = CrosstalkAnalyzer()
        result = ct.analyze(simple_design.wdm_channels)
        assert result["n_channels"] == 4
        assert "worst_xt_db" in result
        assert result["worst_xt_db"] < 0.0

    def test_per_channel_osnr(self, simple_design: PhotonicCircuitDesign) -> None:
        """Each channel reports its OSNR."""
        result = CrosstalkAnalyzer().analyze(simple_design.wdm_channels)
        for ch in result["per_channel"]:
            assert "osnr_db" in ch

    def test_empty_channel_list_has_zero_worst_crosstalk(self) -> None:
        """No channels give a worst crosstalk of 0."""
        result = CrosstalkAnalyzer().analyze([])
        assert result["n_channels"] == 0
        assert result["worst_xt_db"] == 0.0
