# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSleepReportGenerator from former test_sleep_optimizer.py

"""Focused suite: TestSleepReportGenerator from former test_sleep_optimizer.py."""

from __future__ import annotations

import unittest
from sc_neurocore.sleep import (
    SleepOptimizer,
    SleepOptimizerConfig,
    SleepReport,
    SleepReportGenerator,
    SleepStage,
)
from tests.sleep_optimizer_support import (
    generate_stage_eeg,
)


class TestSleepReportGenerator(unittest.TestCase):
    """The sleep report: quality, grade and efficiency of a session."""

    def _run_session(self, protocol: str = "insomnia_relief", n_epochs: int = 50) -> SleepOptimizer:
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

    def test_generate(self) -> None:
        """A session report has a positive duration."""
        r = SleepReportGenerator().generate(self._run_session())
        self.assertIsInstance(r, SleepReport)
        self.assertGreater(r.total_duration_min, 0)

    def test_quality_range(self) -> None:
        """The quality score lies between 0 and 100."""
        r = SleepReportGenerator().generate(self._run_session())
        self.assertGreaterEqual(r.quality_score, 0)
        self.assertLessEqual(r.quality_score, 100)

    def test_grade(self) -> None:
        """The grade is one of A, B, C, D and F."""
        r = SleepReportGenerator().generate(self._run_session())
        self.assertIn(r.grade, "ABCDF")

    def test_efficiency(self) -> None:
        """The sleep efficiency lies between 0 and 100 per cent."""
        r = SleepReportGenerator().generate(self._run_session())
        self.assertGreaterEqual(r.sleep_efficiency_pct, 0)
        self.assertLessEqual(r.sleep_efficiency_pct, 100)

    def test_empty_session(self) -> None:
        """An empty session reports zero duration."""
        cfg = SleepOptimizerConfig(sample_rate=256, fft_window=256, stage_check_interval=256)
        opt = SleepOptimizer("insomnia_relief", config=cfg)
        opt.start_session()
        opt.stop_session()
        r = SleepReportGenerator().generate(opt)
        self.assertEqual(r.total_duration_min, 0.0)
