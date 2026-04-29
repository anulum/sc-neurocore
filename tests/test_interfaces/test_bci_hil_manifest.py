# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for BCI HIL reference manifests

"""Tests for closed-loop BCI HIL reference manifests."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.interfaces import (
    available_bci_hil_profiles,
    build_bci_hil_reference_manifest,
    create_bci_hil_template,
    get_bci_hil_profile,
)


def test_available_bci_hil_profiles_are_deterministic() -> None:
    profiles = available_bci_hil_profiles()

    assert [profile.profile_id for profile in profiles] == ["probe_384ch", "pynq_shd"]
    assert all(profile.safety_contract["hardware_required"] is False for profile in profiles)


def test_get_bci_hil_profile_normalises_identifier() -> None:
    profile = get_bci_hil_profile("PYNQ-SHD")

    assert profile.profile_id == "pynq_shd"
    assert profile.n_channels == 700
    assert profile.model_reference.endswith("shd_speech_classifier")
    assert "aer_payload_generation" in profile.pipeline_steps


def test_get_bci_hil_profile_rejects_unknown_identifier() -> None:
    with pytest.raises(KeyError, match="unknown BCI HIL profile"):
        get_bci_hil_profile("unknown-board")


def test_build_bci_hil_reference_manifest_is_deterministic() -> None:
    manifest = build_bci_hil_reference_manifest("probe_384ch")

    assert list(manifest) == ["schema_version", "profile", "template_config"]
    assert manifest["schema_version"] == "1.0"
    assert manifest["profile"]["profile_id"] == "probe_384ch"
    assert manifest["profile"]["n_channels"] == 384
    assert manifest["template_config"]["waveform_mode"] == "spike"
    assert manifest["template_config"]["input_layer_id"] == "probe_384ch_input"


def test_create_bci_hil_template_processes_probe_window() -> None:
    template = create_bci_hil_template("probe_384ch")
    waveform = np.zeros((96, 384), dtype=np.float32)
    waveform[12, 0] = -24.0
    waveform[48, 10] = -28.0
    waveform[80, 383] = -32.0

    result = template.process_window(waveform, window_start_us=1_000)

    assert result.waveform.n_channels == 384
    assert int(result.spike_raster.sum()) == 3
    assert result.aer.n_events == 3
    assert result.feedback.timestamp_us == 1_000
    assert result.feedback.active_count == 3
    assert result.telemetry["layers"]["probe_384ch_input"]["spike_count"] == 3
    assert result.telemetry["layers"]["probe_384ch_feedback"]["spike_count"] == 3


def test_create_bci_hil_template_uses_shd_channel_count() -> None:
    template = create_bci_hil_template("pynq_shd")

    assert template.config.n_channels == 700
    assert template.config.input_layer_id == "pynq_shd_input"
    assert template.config.feedback_layer_id == "pynq_shd_feedback"
