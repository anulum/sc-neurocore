# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-control bridge compatibility tests

"""Tests for the legacy quantum-control bridge import surface."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import scpneurocore.bridge as bridge
from scpneurocore.bridge import (
    QPU_ARTIFACT_SCHEMA_VERSION,
    QPUBridgeArtifact,
    SourceDataUnavailable,
    load_connectome,
    load_live_stream,
    load_power_grid,
    load_tokamak_data,
    validate_qpu_artifact_payload,
)


def _validate_qpu_payload(payload: dict) -> None:
    validator = getattr(bridge, "validate_qpu_artifact_payload", None)
    assert callable(validator)
    validator(payload)


def _refresh_artifact_hash(payload: dict) -> dict:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(body, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _assert_qpu_artifact(artifact: QPUBridgeArtifact, n: int, mode: str) -> None:
    payload = artifact.to_qpu_artifact_dict()

    assert payload["schema_version"] == QPU_ARTIFACT_SCHEMA_VERSION
    assert artifact.source_mode == mode
    assert artifact.K_nm.shape == (n, n)
    assert artifact.omega.shape == (n,)
    assert artifact.theta0 is None or artifact.theta0.shape == (n,)
    assert len(artifact.layer_assignments) == n
    assert np.all(np.isfinite(artifact.K_nm))
    assert np.all(np.isfinite(artifact.omega))
    assert np.allclose(artifact.K_nm, artifact.K_nm.T)
    assert np.allclose(np.diag(artifact.K_nm), 0.0)
    assert np.all(artifact.K_nm >= 0.0)
    assert payload["hashes"]["K_nm_sha256"]
    assert payload["hashes"]["omega_sha256"]
    assert payload["artifact_sha256"]


def test_expected_import_surface() -> None:
    assert callable(load_connectome)
    assert callable(load_tokamak_data)
    assert callable(load_power_grid)
    assert callable(load_live_stream)
    assert callable(validate_qpu_artifact_payload)


def test_default_loaders_do_not_silently_generate_data() -> None:
    with pytest.raises(SourceDataUnavailable, match="source_mode"):
        load_connectome("c_elegans_sub", n=14)

    with pytest.raises(SourceDataUnavailable, match="source_mode"):
        load_tokamak_data()

    with pytest.raises(SourceDataUnavailable, match="source_mode"):
        load_power_grid(16)

    with pytest.raises(SourceDataUnavailable, match="source_mode"):
        load_live_stream(source="eeg_powergrid", step=0)


def test_publication_source_modes_raise_when_source_is_missing() -> None:
    with pytest.raises(SourceDataUnavailable, match="c_elegans_sub"):
        load_connectome("c_elegans_sub", n=14, source_mode="curated")

    with pytest.raises(SourceDataUnavailable, match="tokamak"):
        load_tokamak_data(source_mode="recorded")


def test_synthetic_connectome_artifact_is_labelled_and_valid() -> None:
    artifact = load_connectome("c_elegans_sub", n=14, source_mode="synthetic")

    _assert_qpu_artifact(artifact, 14, "synthetic")
    assert artifact.domain == "connectome"
    assert artifact.metadata["publication_safe"] is False


def test_synthetic_tokamak_artifact_is_qpu_ready() -> None:
    artifact = load_tokamak_data(n=16, synthetic=True)

    _assert_qpu_artifact(artifact, 16, "synthetic")
    assert artifact.domain == "tokamak"


def test_synthetic_power_grid_artifact_supports_campaign_sizes() -> None:
    for n in (16, 20):
        artifact = load_power_grid(n=n, name="power_grid_europe", source_mode="fixture")
        _assert_qpu_artifact(artifact, n, "fixture")
        assert artifact.domain == "power_grid"


def test_synthetic_live_stream_is_replayable_per_step() -> None:
    a = load_live_stream(source="eeg_powergrid", step=3, source_mode="synthetic")
    b = load_live_stream(source="eeg_powergrid", step=3, source_mode="synthetic")
    c = load_live_stream(source="eeg_powergrid", step=4, source_mode="synthetic")

    _assert_qpu_artifact(a, 12, "synthetic")
    np.testing.assert_allclose(a.K_nm, b.K_nm)
    np.testing.assert_allclose(a.omega, b.omega)
    assert not np.allclose(a.omega, c.omega)
    assert a.replay_id == "synthetic:eeg_powergrid:step:3"


def test_qpu_artifact_hash_rejects_non_finite_json_metadata() -> None:
    artifact = QPUBridgeArtifact(
        domain="power_grid",
        source_name="unit-grid",
        source_mode="fixture",
        K_nm=np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
        omega=np.ones(2, dtype=np.float64),
        theta0=np.zeros(2, dtype=np.float64),
        layer_assignments=[0, 1],
        normalization="unit",
        extraction_method="unit_fixture",
        replay_id="fixture:unit-grid:n2",
        metadata={"bad": float("nan")},
    )

    with pytest.raises(ValueError, match="strict finite JSON"):
        artifact.to_qpu_artifact_dict()


def test_qpu_artifact_rejects_empty_identity_and_provenance_fields() -> None:
    base = {
        "domain": "power_grid",
        "source_name": "unit-grid",
        "source_mode": "fixture",
        "K_nm": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
        "omega": np.ones(2, dtype=np.float64),
        "theta0": np.zeros(2, dtype=np.float64),
        "layer_assignments": [0, 1],
        "normalization": "unit",
        "extraction_method": "unit_fixture",
        "replay_id": "fixture:unit-grid:n2",
    }
    for key in ("domain", "source_name", "normalization", "extraction_method"):
        bad = dict(base)
        bad[key] = ""

        with pytest.raises(ValueError, match=key):
            QPUBridgeArtifact(**bad)


def test_qpu_artifact_rejects_invalid_layer_assignments() -> None:
    base = {
        "domain": "power_grid",
        "source_name": "unit-grid",
        "source_mode": "fixture",
        "K_nm": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
        "omega": np.ones(2, dtype=np.float64),
        "theta0": np.zeros(2, dtype=np.float64),
        "normalization": "unit",
        "extraction_method": "unit_fixture",
        "replay_id": "fixture:unit-grid:n2",
    }
    bad_assignments = ([0, 0], [0, -1], [0, 1.5], [True, 1])

    for layer_assignments in bad_assignments:
        with pytest.raises(ValueError, match="layer_assignments"):
            QPUBridgeArtifact(**base, layer_assignments=layer_assignments)


def test_validate_qpu_artifact_payload_accepts_round_trip_artifact() -> None:
    payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()

    _validate_qpu_payload(payload)


def test_validate_qpu_artifact_payload_rejects_tampered_artifact_hash() -> None:
    payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()
    payload["source_name"] = "tampered-grid"

    with pytest.raises(ValueError, match="artifact_sha256"):
        _validate_qpu_payload(payload)


def test_validate_qpu_artifact_payload_rejects_malformed_arrays_and_hashes() -> None:
    payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()
    bad_cases = [
        ("K_nm", {"K_nm": [[0.0, 0.5], [0.5, 0.0], [0.0, 0.0]]}),
        ("omega", {"omega": [1.0, 1.0, 1.0]}),
        ("theta0", {"theta0": [0.0, 0.0, 0.0]}),
        ("layer_assignments", {"layer_assignments": [0, 1, 1, 3]}),
        ("K_nm_sha256", {"hashes": {**payload["hashes"], "K_nm_sha256": "0" * 64}}),
        ("theta0_sha256", {"hashes": {**payload["hashes"], "theta0_sha256": "f" * 64}}),
    ]

    for match, overrides in bad_cases:
        bad_payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()
        bad_payload.update(overrides)
        _refresh_artifact_hash(bad_payload)

        with pytest.raises(ValueError, match=match):
            _validate_qpu_payload(bad_payload)


def test_validate_qpu_artifact_payload_rejects_malformed_metadata_and_source() -> None:
    payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()
    bad_cases = [
        ("schema_version", {"schema_version": "old"}),
        ("source_mode", {"source_mode": "unknown"}),
        ("source_name", {"source_name": ""}),
        ("metadata", {"metadata": ["not", "mapping"]}),
        ("strict finite JSON", {"metadata": {"bad": float("nan")}}),
        ("artifact_sha256", {"artifact_sha256": "not-a-sha256"}),
    ]

    for match, overrides in bad_cases:
        bad_payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()
        bad_payload.update(overrides)
        if match not in {"artifact_sha256", "strict finite JSON"}:
            _refresh_artifact_hash(bad_payload)

        with pytest.raises(ValueError, match=match):
            _validate_qpu_payload(bad_payload)


def test_bridge_rejects_invalid_source_inputs() -> None:
    with pytest.raises(ValueError, match="unsupported connectome"):
        load_connectome("unknown", source_mode="synthetic")

    with pytest.raises(ValueError, match="unsupported live stream"):
        load_live_stream(source="unknown", step=0, source_mode="synthetic")

    with pytest.raises(ValueError, match="step"):
        load_live_stream(source="eeg_powergrid", step=-1, source_mode="synthetic")
