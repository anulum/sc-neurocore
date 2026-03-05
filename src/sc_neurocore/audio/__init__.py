# SPDX-License-Identifier: AGPL-3.0-or-later
"""
SSGF Adaptive Audio Engine
============================

Lightweight SSGF geometry solver, Entrainment Verification Score (EVS),
and closed-loop adaptive audio controller for sc-neurocore.

Modules
-------
ssgf_engine   -- Stochastic Synthesis of Geometric Fields (Kuramoto + geometry)
evs_engine    -- Entrainment Verification Score (FFT-based EEG scoring)
adaptive_engine -- Closed-loop adaptive audio controller (SSGF + EVS)
user_profile  -- User chronotype and session preference model
"""

from .ssgf_engine import SSGFConfig, SSGFEngine
from .evs_engine import EVSConfig, EVSEngine, EVSSnapshot
from .adaptive_engine import (
    AdaptiveAudioEngine,
    AdaptiveSessionReport,
    SessionPhase,
)
from .user_profile import UserProfile, Chronotype

__all__ = [
    "SSGFConfig",
    "SSGFEngine",
    "EVSConfig",
    "EVSEngine",
    "EVSSnapshot",
    "AdaptiveAudioEngine",
    "AdaptiveSessionReport",
    "SessionPhase",
    "UserProfile",
    "Chronotype",
]
