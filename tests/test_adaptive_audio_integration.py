# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIntegration from former test_adaptive_audio.py

"""Focused suite: TestIntegration from former test_adaptive_audio.py."""

from __future__ import annotations

import unittest
from sc_neurocore.audio import (
    AdaptiveAudioEngine,
    Chronotype,
    EVSEngine,
    EVSSnapshot,
    SSGFEngine,
    UserProfile,
)
from sc_neurocore.audio.ssgf_engine import (
    SSGFConfig,
)


class TestIntegration(unittest.TestCase):
    """The adaptive engine, SSGF engine, EVS engine and user profile together."""

    def test_full_pipeline(self) -> None:
        """Twenty EVS updates return binaural frequencies and a report with ticks."""
        profile = UserProfile(chronotype=Chronotype.BEAR)
        ssgf = SSGFEngine()
        evs = EVSEngine()
        adaptive = AdaptiveAudioEngine(ssgf, evs, profile)

        snap = EVSSnapshot(
            evs_score=55.0,
            relative_increase=0.4,
            peak_alignment=0.6,
            band_dominance=0.25,
            temporal_consistency=0.7,
            is_verified=True,
            confidence=0.65,
            target_hz=10.0,
            peak_hz=10.5,
            band_powers={"alpha": 0.4},
            timestamp=0,
        )
        for _ in range(20):
            result = adaptive.on_evs_update(snap)
            self.assertIn("binaural_hz", result)

        report = adaptive.get_session_report()
        self.assertGreater(report.total_ticks, 0)

    def test_ssgf_audio_consistency(self) -> None:
        """After five outer steps the audio mapping keeps its frequency and intensity in range."""
        eng = SSGFEngine(SSGFConfig(seed=42))
        for _ in range(5):
            eng.outer_step()
        m = eng.get_audio_mapping()
        self.assertGreaterEqual(m["binaural_hz"], 0.5)
        self.assertLessEqual(m["binaural_hz"], 40.0)
        self.assertGreaterEqual(m["intensity"], 0.0)
        self.assertLessEqual(m["intensity"], 1.0)
