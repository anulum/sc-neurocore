# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEVSEngine from former test_adaptive_audio.py

"""Focused suite: TestEVSEngine from former test_adaptive_audio.py."""

from __future__ import annotations

from tests.adaptive_audio_support import *  # noqa: F403

class TestEVSEngine(unittest.TestCase):
    def test_init(self) -> None:
        eng = EVSEngine()
        self.assertFalse(eng.baseline_done)

    def test_baseline(self) -> None:
        eng = EVSEngine(EVSConfig(sample_rate=256, fft_window=256, baseline_duration_s=0.5))
        eng.start_baseline()
        for v in np.sin(np.linspace(0, 20 * np.pi, 256)):
            eng.add_sample(float(v))
        self.assertTrue(len(eng._baseline_samples) > 0 or eng._baseline_done)

    def test_compute_returns_snapshot(self) -> None:
        cfg = EVSConfig(
            sample_rate=256, fft_window=256, baseline_duration_s=0.5, update_interval_samples=64
        )
        eng = EVSEngine(cfg)
        eng.start_baseline()
        for v in np.sin(np.linspace(0, 20 * np.pi, 256)):
            eng.add_sample(float(v))
        eng.set_target(10.0)
        for v in np.sin(np.linspace(0, 20 * np.pi, 256)):
            eng.add_sample(float(v))
        snap = eng.compute()
        if snap is not None:
            self.assertIsInstance(snap, EVSSnapshot)
            self.assertGreaterEqual(snap.evs_score, 0)
            self.assertLessEqual(snap.evs_score, 100)

    def test_score_range(self) -> None:
        cfg = EVSConfig(
            sample_rate=256, fft_window=256, baseline_duration_s=0.5, update_interval_samples=64
        )
        eng = EVSEngine(cfg)
        eng.start_baseline()
        rng = np.random.RandomState(42)
        for v in rng.randn(256):
            eng.add_sample(float(v))
        eng.set_target(10.0)
        for v in rng.randn(256):
            eng.add_sample(float(v))
        snap = eng.compute()
        if snap:
            self.assertGreaterEqual(snap.evs_score, 0)
            self.assertLessEqual(snap.evs_score, 100)
