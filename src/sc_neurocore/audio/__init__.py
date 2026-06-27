# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SSGF Adaptive Audio Engine

"""Package facade for the SSGF adaptive audio engine.

Lightweight SSGF geometry solver, Entrainment Verification Score (EVS),
and closed-loop adaptive audio controller for sc-neurocore.

Modules
-------
ssgf_engine -- Stochastic Synthesis of Geometric Fields (Kuramoto + geometry)
evs_engine -- Entrainment Verification Score (FFT-based EEG scoring)
adaptive_engine -- Closed-loop adaptive audio controller (SSGF + EVS)
user_profile -- User chronotype and session preference model
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
