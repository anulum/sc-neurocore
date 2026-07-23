# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAmplitudeCutoff from former test_spike_stats_sorting_quality.py

"""Focused suite: TestAmplitudeCutoff from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403

class TestAmplitudeCutoff:
    def test_typical(self) -> None:
        rng = _rng()
        amps = rng.normal(1.0, 0.3, 200)
        result = amplitude_cutoff(amps)
        assert 0 <= result <= 1

    def test_too_few(self) -> None:
        result = amplitude_cutoff(np.array([1.0, 2.0]))
        assert np.isnan(result)

    def test_peak_at_zero(self) -> None:
        # Force peak_idx == 0 by having most amplitudes near zero
        amps = np.concatenate([np.zeros(90), np.array([1.0] * 10)])
        result = amplitude_cutoff(amps)
        assert result == 0.5

    def test_all_zero_amplitudes(self) -> None:
        # All-identical amplitudes collapse into bin 0 (peak_idx == 0) and return
        # the 0.5 sentinel — a finite, well-defined degenerate result.
        amps = np.zeros(20)
        result = amplitude_cutoff(amps)
        assert result == 0.5
