# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sleep_optimizer.py

from __future__ import annotations

"""Tests for Sleep Optimization System (UC3) — adapts to agent-written API."""
import unittest
from unittest import mock
import numpy as np
from sc_neurocore.sleep import (
    SleepStageDetector,
    SleepStage,
    CircadianOptimizer,
    Chronotype,
    get_protocol,
    list_protocols,
    SleepOptimizer,
    SleepOptimizerConfig,
    SleepReportGenerator,
    SleepReport,
)
from sc_neurocore.sleep.sleep_stage_detector import DetectorConfig, STAGE_SIGNATURES, EEG_BANDS
from sc_neurocore.sleep.protocol_library import StageAudioParams, PROTOCOL_REGISTRY
from sc_neurocore.sleep.sleep_optimizer import SleepTick
def generate_stage_eeg(stage, sample_rate=256, n_samples=512, seed=42):
    t = np.arange(n_samples) / sample_rate
    rng = np.random.RandomState(seed)
    if stage == SleepStage.WAKE:
        signal = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
    elif stage == SleepStage.N1:
        signal = 0.6 * np.sin(2 * np.pi * 6 * t) + 0.2 * np.sin(2 * np.pi * 10 * t)
    elif stage == SleepStage.N2:
        signal = 0.5 * np.sin(2 * np.pi * 3 * t) + 0.4 * np.sin(2 * np.pi * 1.5 * t)
    elif stage == SleepStage.N3:
        signal = 0.8 * np.sin(2 * np.pi * 1.5 * t) + 0.3 * np.sin(2 * np.pi * 0.8 * t)
    elif stage == SleepStage.REM:
        signal = 0.4 * np.sin(2 * np.pi * 6 * t) + 0.3 * np.sin(2 * np.pi * 15 * t)
    else:
        signal = np.zeros(n_samples)
    return signal + rng.normal(0, 0.15, n_samples)
if __name__ == "__main__":
    unittest.main()

__all__ = ['unittest', 'mock', 'np', 'SleepStageDetector', 'SleepStage', 'CircadianOptimizer', 'Chronotype', 'get_protocol', 'list_protocols', 'SleepOptimizer', 'SleepOptimizerConfig', 'SleepReportGenerator', 'SleepReport', 'DetectorConfig', 'STAGE_SIGNATURES', 'EEG_BANDS', 'StageAudioParams', 'PROTOCOL_REGISTRY', 'SleepTick', 'generate_stage_eeg']
