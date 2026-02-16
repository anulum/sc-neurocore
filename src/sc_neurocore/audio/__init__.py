"""
SSGF Adaptive Audio Package
============================

Real-time adaptive brainwave entrainment using SSGF geometry evolution
with EVS (Entrainment Verification Score) feedback.

Modules:
    ssgf_engine      - SSGF geometry solver (latent z → W → spectral)
    evs_engine        - Entrainment verification scoring
    adaptive_engine   - Feedback loop: EVS → SSGF adaptation → audio params
    user_profile      - User priors and personalization

Author: Claude (Session 2026-02-16)
"""

from .ssgf_engine import SSGFEngine, SSGFConfig, SSGFState
from .evs_engine import EVSEngine, EVSConfig, EVSSnapshot
from .adaptive_engine import AdaptiveAudioEngine, AdaptiveConfig
from .user_profile import UserProfile, Chronotype

__all__ = [
    "SSGFEngine",
    "SSGFConfig",
    "SSGFState",
    "EVSEngine",
    "EVSConfig",
    "EVSSnapshot",
    "AdaptiveAudioEngine",
    "AdaptiveConfig",
    "UserProfile",
    "Chronotype",
]
