# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSleepOptimizer from former test_sleep_optimizer.py

"""Focused suite: TestSleepOptimizer from former test_sleep_optimizer.py."""

from __future__ import annotations

import unittest
from sc_neurocore.sleep import (
    SleepOptimizer,
    SleepOptimizerConfig,
    SleepStage,
)
from sc_neurocore.sleep.protocol_library import (
    StageAudioParams,
)
from tests.sleep_optimizer_support import (
    generate_stage_eeg,
)


class TestSleepOptimizer(unittest.TestCase):
    """The sleep optimiser's sessions, ticks, audio parameters and stage durations."""

    def setUp(self) -> None:
        """Create the shared optimiser configuration."""
        self.cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)

    def test_start(self) -> None:
        """A session starts (the test asserts nothing)."""
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()

    def test_protocol_from_string(self) -> None:
        """Protocol from string."""
        opt = SleepOptimizer("deep_sleep_boost", config=self.cfg)
        opt.start_session()
        self.assertEqual(opt.protocol.name, "deep_sleep_boost")

    def test_check_none_without_samples(self) -> None:
        """Check none without samples."""
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_sample(0.1)
        self.assertIsNone(opt.check_and_adapt())

    def test_check_with_samples(self) -> None:
        """With enough samples a check returns a tick."""
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256))
        result = opt.check_and_adapt()
        self.assertIsNotNone(result)

    def test_tick_increments(self) -> None:
        """Three checks give tick 3."""
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        for i in range(3):
            opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i))
            r = opt.check_and_adapt()
        assert r is not None
        self.assertEqual(r.tick, 3)

    def test_audio_params(self) -> None:
        """A tick carries stage audio parameters with a positive binaural frequency."""
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256))
        r = opt.check_and_adapt()
        assert r is not None
        # audio_params is a StageAudioParams dataclass
        self.assertIsInstance(r.audio_params, StageAudioParams)
        self.assertGreater(r.audio_params.binaural_hz, 0)

    def test_stage_durations(self) -> None:
        """Stage durations sum to more than zero after a session."""
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        for i in range(5):
            opt.add_samples(generate_stage_eeg(SleepStage.N3, n_samples=256, seed=i))
            opt.check_and_adapt()
        self.assertGreater(sum(opt.get_stage_durations().values()), 0)

    def test_history(self) -> None:
        """Three checks leave three ticks in the history."""
        opt = SleepOptimizer("deep_sleep_boost", config=self.cfg)
        opt.start_session()
        for i in range(3):
            opt.add_samples(generate_stage_eeg(SleepStage.N3, n_samples=256, seed=i))
            opt.check_and_adapt()
        self.assertEqual(len(opt.get_history()), 3)

    def test_hypnogram(self) -> None:
        """A session yields a non-empty hypnogram."""
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        for i in range(5):
            opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i))
            opt.check_and_adapt()
        self.assertGreater(len(opt.get_hypnogram()), 0)

    def test_stop(self) -> None:
        """Stopping a session returns its history list."""
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        result = opt.stop_session()
        self.assertIsInstance(result, list)

    def test_state(self) -> None:
        """The state dictionary names the protocol."""
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.REM, n_samples=256))
        opt.check_and_adapt()
        s = opt.get_state()
        self.assertIsInstance(s, dict)
        self.assertIn("protocol", s)

    def test_tick_has_current_stage(self) -> None:
        """Tick has current stage."""
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256))
        tick = opt.check_and_adapt()
        assert tick is not None
        self.assertIn(tick.current_stage, list(SleepStage))
