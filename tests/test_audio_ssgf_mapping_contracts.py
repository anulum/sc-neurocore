# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SSGF audio mapping contracts

"""Contracts for SSGF small-N audio mapping fallback branches."""

from __future__ import annotations

from sc_neurocore.audio.ssgf_engine import SSGFConfig, SSGFEngine


def test_ssgf_n2_mapping_uses_all_small_network_fallbacks() -> None:
    engine = SSGFEngine(SSGFConfig(N=2))

    engine.outer_step()
    mapping = engine.get_audio_mapping()

    assert mapping["binaural_hz"] == 10.0
    assert mapping["pulse_rate"] == 8.0
    assert mapping["spatial_angle"] == 0.0


def test_ssgf_n3_mapping_preserves_pulse_and_spatial_fallbacks() -> None:
    engine = SSGFEngine(SSGFConfig(N=3))

    engine.outer_step()
    mapping = engine.get_audio_mapping()

    assert mapping["pulse_rate"] == 8.0
    assert mapping["spatial_angle"] == 0.0


def test_ssgf_n5_mapping_preserves_spatial_fallback() -> None:
    engine = SSGFEngine(SSGFConfig(N=5))

    engine.outer_step()
    mapping = engine.get_audio_mapping()

    assert mapping["spatial_angle"] == 0.0
