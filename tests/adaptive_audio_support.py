# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_adaptive_audio.py

from __future__ import annotations

"""Tests for SSGF Adaptive Audio (UC1) — 32 tests."""
import unittest
import numpy as np
from sc_neurocore.audio import (
    SSGFEngine,
    EVSEngine,
    EVSSnapshot,
    AdaptiveAudioEngine,
    AdaptiveSessionReport,
    UserProfile,
    Chronotype,
)
from sc_neurocore.audio.ssgf_engine import SSGFConfig
from sc_neurocore.audio.evs_engine import EVSConfig
from sc_neurocore.audio.adaptive_engine import SessionPhase


def _snapshot(
    *,
    evs_score: float = 60.0,
    is_verified: bool = True,
    peak_alignment: float = 0.7,
    target_hz: float = 10.0,
    peak_hz: float = 10.2,
) -> EVSSnapshot:
    return EVSSnapshot(
        evs_score=evs_score,
        relative_increase=0.5,
        peak_alignment=peak_alignment,
        band_dominance=0.3,
        temporal_consistency=0.8,
        is_verified=is_verified,
        confidence=0.7,
        target_hz=target_hz,
        peak_hz=peak_hz,
        band_powers={"alpha": 0.5},
        timestamp=0,
    )


def _small_adaptive_engine() -> AdaptiveAudioEngine:
    ssgf = SSGFEngine(SSGFConfig(N=4, z_dim=6, micro_steps=1, seed=7))
    return AdaptiveAudioEngine(ssgf, EVSEngine(), UserProfile())


if __name__ == "__main__":
    unittest.main()

__all__ = [
    "unittest",
    "np",
    "SSGFEngine",
    "EVSEngine",
    "EVSSnapshot",
    "AdaptiveAudioEngine",
    "AdaptiveSessionReport",
    "UserProfile",
    "Chronotype",
    "SSGFConfig",
    "EVSConfig",
    "SessionPhase",
    "_snapshot",
    "_small_adaptive_engine",
]
