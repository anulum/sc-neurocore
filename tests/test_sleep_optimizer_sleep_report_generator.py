# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSleepReportGenerator from former test_sleep_optimizer.py

"""Focused suite: TestSleepReportGenerator from former test_sleep_optimizer.py."""

from __future__ import annotations

from tests.sleep_optimizer_support import *  # noqa: F403


class TestSleepReportGenerator(unittest.TestCase):
    def _run_session(self, protocol="insomnia_relief", n_epochs=50):
        cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)
        opt = SleepOptimizer(protocol, config=cfg)
        opt.start_session()
        stages = [
            SleepStage.WAKE,
            SleepStage.N1,
            SleepStage.N2,
            SleepStage.N3,
            SleepStage.N2,
            SleepStage.REM,
            SleepStage.N1,
        ]
        for i in range(n_epochs):
            stage = stages[min(int(i / n_epochs * len(stages)), len(stages) - 1)]
            opt.add_samples(generate_stage_eeg(stage, n_samples=256, seed=i))
            opt.check_and_adapt()
        opt.stop_session()
        return opt

    def test_generate(self):
        r = SleepReportGenerator().generate(self._run_session())
        self.assertIsInstance(r, SleepReport)
        self.assertGreater(r.total_duration_min, 0)

    def test_quality_range(self):
        r = SleepReportGenerator().generate(self._run_session())
        self.assertGreaterEqual(r.quality_score, 0)
        self.assertLessEqual(r.quality_score, 100)

    def test_grade(self):
        r = SleepReportGenerator().generate(self._run_session())
        self.assertIn(r.grade, "ABCDF")

    def test_efficiency(self):
        r = SleepReportGenerator().generate(self._run_session())
        self.assertGreaterEqual(r.sleep_efficiency_pct, 0)
        self.assertLessEqual(r.sleep_efficiency_pct, 100)

    def test_empty_session(self):
        cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)
        opt = SleepOptimizer("insomnia_relief", config=cfg)
        opt.start_session()
        opt.stop_session()
        r = SleepReportGenerator().generate(opt)
        self.assertEqual(r.total_duration_min, 0.0)
