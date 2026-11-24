# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSleepStageDetector from former test_sleep_optimizer.py

"""Focused suite: TestSleepStageDetector from former test_sleep_optimizer.py."""

from __future__ import annotations

import numpy as np
import unittest
from sc_neurocore.sleep import (
    SleepStage,
    SleepStageDetector,
)
from sc_neurocore.sleep.sleep_stage_detector import (
    DetectorConfig,
    EEG_BANDS,
    STAGE_SIGNATURES,
)
from tests.sleep_optimizer_support import (
    generate_stage_eeg,
)


class TestSleepStageDetector(unittest.TestCase):
    """The EEG sleep-stage detector and its band powers."""

    def test_default_construction(self) -> None:
        """Default construction."""
        det = SleepStageDetector()
        self.assertIsNotNone(det.config)

    def test_detect_without_enough_samples(self) -> None:
        """Detect without enough samples."""
        det = SleepStageDetector(DetectorConfig(min_samples=128))
        for _ in range(50):
            det.add_sample(0.0)
        self.assertIsNone(det.detect())

    def test_add_samples_bulk(self) -> None:
        """Adding 100 samples at once fills the buffer with 100."""
        det = SleepStageDetector()
        det.add_samples(np.zeros(100))
        self.assertEqual(len(det._buffer), 100)

    def test_detect_wake(self) -> None:
        """Wake-like EEG is detected as wake, N1 or REM."""
        det = SleepStageDetector(DetectorConfig(sample_rate=256, fft_window=512))
        det.add_samples(generate_stage_eeg(SleepStage.WAKE, n_samples=512))
        result = det.detect()
        self.assertIsNotNone(result)
        self.assertIn(result, [SleepStage.WAKE, SleepStage.N1, SleepStage.REM])

    def test_detect_n3(self) -> None:
        """Deep-sleep EEG is detected as N3 or N2."""
        det = SleepStageDetector(DetectorConfig(sample_rate=256, fft_window=512))
        det.add_samples(generate_stage_eeg(SleepStage.N3, n_samples=512))
        result = det.detect()
        self.assertIsNotNone(result)
        self.assertIn(result, [SleepStage.N3, SleepStage.N2])

    def test_band_powers_computed(self) -> None:
        """Band powers computed."""
        det = SleepStageDetector(DetectorConfig(sample_rate=256, fft_window=256))
        det.add_samples(generate_stage_eeg(SleepStage.WAKE, n_samples=256))
        det.detect()
        powers = det.get_band_powers()
        assert powers is not None
        for band in ["alpha", "beta", "delta", "theta", "gamma"]:
            self.assertIn(band, powers)
            self.assertGreaterEqual(powers[band], 0.0)

    def test_classify_static(self) -> None:
        """Static classification of wake band powers gives wake."""
        # _classify is a static method taking a numpy power vector
        power_vec = np.array([0.05, 0.10, 0.35, 0.35, 0.15])  # WAKE signature
        stage = SleepStageDetector._classify(power_vec)
        self.assertEqual(stage, SleepStage.WAKE)

    def test_classify_n3(self) -> None:
        """Static classification of deep-sleep band powers gives N3."""
        power_vec = np.array([0.60, 0.20, 0.10, 0.07, 0.03])  # N3 signature
        stage = SleepStageDetector._classify(power_vec)
        self.assertEqual(stage, SleepStage.N3)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        det = SleepStageDetector()
        det.add_samples(np.ones(100))
        det.detect()
        det.reset()
        self.assertEqual(len(det._buffer), 0)

    def test_stage_signatures_complete(self) -> None:
        """Stage signatures complete."""
        for stage in SleepStage:
            self.assertIn(stage, STAGE_SIGNATURES)

    def test_eeg_bands_defined(self) -> None:
        """Every expected EEG band is defined."""
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            self.assertIn(band, EEG_BANDS)
