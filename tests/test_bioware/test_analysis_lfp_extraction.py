# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLFPExtraction from former test_analysis.py

"""Focused suite: TestLFPExtraction from former test_analysis.py."""

from __future__ import annotations

from tests.test_bioware.analysis_support import *  # noqa: F403


class TestLFPExtraction:
    def test_default_bands(self) -> None:
        data = _synth_voltage(n_samples=2000, n_channels=5)
        result = extract_lfp_power(data, sample_rate_hz=20000.0)
        assert "delta" in result
        assert "gamma" in result
        assert result["delta"].shape == (5,)

    def test_custom_band(self) -> None:
        data = _synth_voltage(n_samples=2000, n_channels=5)
        bands = [LFPBand("custom", 10.0, 50.0)]
        result = extract_lfp_power(data, sample_rate_hz=20000.0, bands=bands)
        assert "custom" in result
        assert np.all(result["custom"] >= 0)
