# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrosstalkAnalyzer from former test_bridges_photonic_noc.py

"""Focused suite: TestCrosstalkAnalyzer from former test_bridges_photonic_noc.py."""

from __future__ import annotations

from tests.bridges_photonic_noc_support import *  # noqa: F403


class TestCrosstalkAnalyzer:
    def test_analyze_returns_result(self):
        analyzer = CrosstalkAnalyzer()
        ch1 = WDMChannel(channel_id=0, wavelength_nm=1550.0, bandwidth_nm=0.4, signal_name="s1")
        ch2 = WDMChannel(channel_id=1, wavelength_nm=1550.8, bandwidth_nm=0.4, signal_name="s2")
        result = analyzer.analyze([ch1, ch2])
        assert isinstance(result, dict)
