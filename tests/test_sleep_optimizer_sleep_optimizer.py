# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSleepOptimizer from former test_sleep_optimizer.py

"""Focused suite: TestSleepOptimizer from former test_sleep_optimizer.py."""

from __future__ import annotations

from tests.sleep_optimizer_support import *  # noqa: F403

class TestSleepOptimizer(unittest.TestCase):
    def setUp(self):
        self.cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)

    def test_start(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()

    def test_protocol_from_string(self):
        opt = SleepOptimizer("deep_sleep_boost", config=self.cfg)
        opt.start_session()
        self.assertEqual(opt.protocol.name, "deep_sleep_boost")

    def test_check_none_without_samples(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_sample(0.1)
        self.assertIsNone(opt.check_and_adapt())

    def test_check_with_samples(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256))
        result = opt.check_and_adapt()
        self.assertIsNotNone(result)

    def test_tick_increments(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        for i in range(3):
            opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i))
            r = opt.check_and_adapt()
        self.assertEqual(r.tick, 3)

    def test_audio_params(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256))
        r = opt.check_and_adapt()
        # audio_params is a StageAudioParams dataclass
        self.assertIsInstance(r.audio_params, StageAudioParams)
        self.assertGreater(r.audio_params.binaural_hz, 0)

    def test_stage_durations(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        for i in range(5):
            opt.add_samples(generate_stage_eeg(SleepStage.N3, n_samples=256, seed=i))
            opt.check_and_adapt()
        self.assertGreater(sum(opt.get_stage_durations().values()), 0)

    def test_history(self):
        opt = SleepOptimizer("deep_sleep_boost", config=self.cfg)
        opt.start_session()
        for i in range(3):
            opt.add_samples(generate_stage_eeg(SleepStage.N3, n_samples=256, seed=i))
            opt.check_and_adapt()
        self.assertEqual(len(opt.get_history()), 3)

    def test_hypnogram(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        for i in range(5):
            opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256, seed=i))
            opt.check_and_adapt()
        self.assertGreater(len(opt.get_hypnogram()), 0)

    def test_stop(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        result = opt.stop_session()
        self.assertIsInstance(result, list)

    def test_state(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.REM, n_samples=256))
        opt.check_and_adapt()
        s = opt.get_state()
        self.assertIsInstance(s, dict)
        self.assertIn("protocol", s)

    def test_tick_has_current_stage(self):
        opt = SleepOptimizer("insomnia_relief", config=self.cfg)
        opt.start_session()
        opt.add_samples(generate_stage_eeg(SleepStage.N2, n_samples=256))
        tick = opt.check_and_adapt()
        self.assertIn(tick.current_stage, list(SleepStage))
