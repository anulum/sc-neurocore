# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Audio package API tests

"""Package-facade tests for the adaptive audio public API."""

from __future__ import annotations

import sc_neurocore.audio as audio
from sc_neurocore.audio.adaptive_engine import (
    AdaptiveAudioEngine,
    AdaptiveSessionReport,
    SessionPhase,
)
from sc_neurocore.audio.evs_engine import EVSConfig, EVSEngine, EVSSnapshot
from sc_neurocore.audio.ssgf_engine import SSGFConfig, SSGFEngine
from sc_neurocore.audio.user_profile import Chronotype, UserProfile


def test_audio_package_exports_adaptive_audio_api() -> None:
    """Package entry point re-exports the documented adaptive-audio API."""
    assert audio.__all__ == [
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
    assert audio.SSGFConfig is SSGFConfig
    assert audio.SSGFEngine is SSGFEngine
    assert audio.EVSConfig is EVSConfig
    assert audio.EVSEngine is EVSEngine
    assert audio.EVSSnapshot is EVSSnapshot
    assert audio.AdaptiveAudioEngine is AdaptiveAudioEngine
    assert audio.AdaptiveSessionReport is AdaptiveSessionReport
    assert audio.SessionPhase is SessionPhase
    assert audio.UserProfile is UserProfile
    assert audio.Chronotype is Chronotype


def test_audio_package_import_runs_ssgf_mapping_step() -> None:
    """Package-level SSGFEngine produces the documented audio mapping."""
    engine = audio.SSGFEngine(audio.SSGFConfig(N=4, z_dim=6, micro_steps=1, seed=11))

    cost = engine.outer_step()
    mapping = engine.get_audio_mapping()

    assert cost > 0.0
    assert set(mapping) == {
        "binaural_hz",
        "fiedler",
        "intensity",
        "pulse_rate",
        "spectral_gap",
        "spatial_angle",
        "theurgic_mode",
    }
