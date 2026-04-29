# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR hardware target manifests

"""Tests for NIR hardware capability manifests and noise annotations."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.nir_bridge import (
    available_hardware_profiles,
    build_nir_hardware_manifest,
    build_noise_annotation,
    get_hardware_profile,
)


def test_available_hardware_profiles_are_deterministic() -> None:
    profiles = available_hardware_profiles()

    assert [profile.target_id for profile in profiles] == [
        "akida",
        "brainscales3",
        "dynap_se",
        "loihi2",
        "spinnaker2",
    ]
    assert all(profile.backend_status == "capability_manifest" for profile in profiles)


def test_get_hardware_profile_normalises_identifier() -> None:
    profile = get_hardware_profile("DYNAP-SE")

    assert profile.target_id == "dynap_se"
    assert profile.display_name == "DYNAP-SE"
    assert "aer_drop_rate" in profile.sc_constraints.back_annotation_channels


def test_akida_profile_is_conservative_manifest_only() -> None:
    profile = get_hardware_profile("akida")

    assert profile.display_name == "Akida"
    assert profile.backend_status == "capability_manifest"
    assert "Conv2d" in profile.supported_nir_nodes
    assert "Delay" in profile.unsupported_nir_nodes
    assert profile.sc_constraints.stream_transport == "event_rate_probability"
    assert "quantisation_error_rate" in profile.sc_constraints.back_annotation_channels


def test_get_hardware_profile_rejects_unknown_identifier() -> None:
    with pytest.raises(KeyError, match="unknown neuromorphic target"):
        get_hardware_profile("unknown-chip")


def test_build_nir_hardware_manifest_filters_targets() -> None:
    manifest = build_nir_hardware_manifest(("loihi2", "spinnaker2"))

    assert manifest["schema_version"] == "1.0"
    assert manifest["extension"] == "sc_neurocore.nir_hardware_targets"
    assert [profile["target_id"] for profile in manifest["profiles"]] == [
        "loihi2",
        "spinnaker2",
    ]
    assert manifest["profiles"][0]["sc_constraints"]["bitstream_lengths"] == [
        64,
        128,
        256,
        512,
        1024,
    ]


def test_noise_annotation_accepts_known_channels() -> None:
    annotation = build_noise_annotation(
        "loihi2",
        {"spike_drop_rate": 0.001, "timing_jitter_ns": 4.5},
    )

    payload = annotation.to_dict()
    assert payload["target_id"] == "loihi2"
    assert payload["observations"] == {"spike_drop_rate": 0.001, "timing_jitter_ns": 4.5}
    assert payload["simulation_contract"]["requires_measured_hardware"] is True


def test_noise_annotation_rejects_unknown_channels() -> None:
    with pytest.raises(ValueError, match="unknown noise channels"):
        build_noise_annotation("spinnaker2", {"threshold_drift": 0.1})


@pytest.mark.parametrize("bad_value", [-1.0, math.inf, math.nan])
def test_noise_annotation_rejects_invalid_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_noise_annotation("brainscales3", {"threshold_drift": bad_value})
