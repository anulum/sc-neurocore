# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sleep edge contracts

"""Contracts for circadian windows, protocol fallback, and zero-signal stage detection."""

from __future__ import annotations

import numpy as np

from sc_neurocore.sleep.circadian_optimizer import Chronotype, CircadianOptimizer
from sc_neurocore.sleep.protocol_library import SleepProtocol
from sc_neurocore.sleep.sleep_stage_detector import SleepStage, SleepStageDetector


def test_circadian_optimizer_handles_wrapping_and_non_wrapping_windows() -> None:
    wolf = CircadianOptimizer(Chronotype.WOLF)
    lion = CircadianOptimizer(Chronotype.LION)

    assert wolf.is_in_sleep_window(4.0) is True
    assert wolf.is_in_sleep_window(12.0) is False
    assert lion.is_in_sleep_window(23.0) is True
    assert lion.is_in_sleep_window(3.0) is True
    assert lion.is_in_sleep_window(12.0) is False


def test_sleep_protocol_defaults_to_rem_when_no_targets_match() -> None:
    protocol = SleepProtocol(stage_targets={})

    assert protocol.get_target_stage(0.5) == SleepStage.REM


def test_sleep_stage_detector_classifies_zero_signal_as_wake() -> None:
    assert SleepStageDetector._classify(np.zeros(5)) == SleepStage.WAKE
