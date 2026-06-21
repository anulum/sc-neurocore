# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio simulation manifest tests

"""Tests for path-free Studio simulation run manifests."""

from __future__ import annotations

import hashlib
import json
import math
import re

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.simulation_manifest import (
    STUDIO_SIMULATION_RUN_SCHEMA_VERSION,
    build_simulation_run_manifest,
)


@pytest.fixture
def client() -> TestClient:
    """Return a Studio test client."""

    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_build_simulation_run_manifest_returns_path_free_hashes() -> None:
    """Simulation manifests describe reproducibility without local paths."""

    request_payload = {
        "equations": ["dv/dt = I"],
        "dt": 0.1,
        "duration": 1.0,
        "current": 1.0,
    }
    result_payload = {
        "time": [0.0, 0.1],
        "states": {"v": [0.0, 0.1]},
        "spikes": [],
        "spike_count": 0,
        "dt": 0.1,
        "n_steps": 10,
        "pattern": {"pattern": "silent", "description": "No spikes detected."},
    }

    manifest = build_simulation_run_manifest(
        source="ode",
        request_payload=request_payload,
        result_payload=result_payload,
    )
    public = manifest.to_public_dict()

    assert public == {
        "dt": 0.1,
        "evidence_classification": "simulation",
        "input_sha256": hashlib.sha256(
            json.dumps(
                request_payload,
                allow_nan=False,
                default=str,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "n_steps": 10,
        "result_sha256": hashlib.sha256(
            json.dumps(
                result_payload,
                allow_nan=False,
                default=str,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "sample_count": 2,
        "schema_version": STUDIO_SIMULATION_RUN_SCHEMA_VERSION,
        "source": "ode",
        "spike_count": 0,
        "state_variables": ["v"],
    }
    assert "path" not in public


def test_simulation_run_manifest_excludes_existing_manifest_from_result_hash() -> None:
    """Result hashes stay stable when callers pass an already annotated result."""

    request_payload = {"name": "AdExNeuron"}
    result_payload = {
        "time": [0.0],
        "states": {"v": [0.0]},
        "spike_count": 0,
        "dt": 0.1,
        "n_steps": 1,
        "run_metadata": {"stale": True},
    }

    manifest = build_simulation_run_manifest(
        source="model",
        request_payload=request_payload,
        result_payload=result_payload,
    )
    expected_payload = {
        "time": [0.0],
        "states": {"v": [0.0]},
        "spike_count": 0,
        "dt": 0.1,
        "n_steps": 1,
    }

    assert manifest.to_public_dict()["source"] == "model"
    assert (
        manifest.to_public_dict()["result_sha256"]
        == hashlib.sha256(
            json.dumps(
                expected_payload,
                allow_nan=False,
                default=str,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"bad": math.nan},
        {"bad": math.inf},
    ],
)
def test_simulation_run_manifest_rejects_non_portable_json(
    payload: dict[str, float],
) -> None:
    """Simulation manifests fail closed on non-portable JSON values."""

    with pytest.raises(ValueError, match="portable JSON"):
        build_simulation_run_manifest(
            source="ode",
            request_payload=payload,
            result_payload={"dt": 0.1, "n_steps": 1, "spike_count": 0},
        )
    with pytest.raises(ValueError, match="portable JSON"):
        build_simulation_run_manifest(
            source="ode",
            request_payload={},
            result_payload=payload,
        )


def test_ode_simulation_endpoint_returns_run_metadata(client: TestClient) -> None:
    """ODE simulation responses include path-free reproducibility metadata."""

    response = client.post(
        "/api/simulate",
        json={
            "equations": ["dv/dt = I"],
            "init": {"v": 0.0},
            "dt": 0.1,
            "duration": 1.0,
            "current": 1.0,
        },
    )

    assert response.status_code == 200
    metadata = response.json()["run_metadata"]
    assert metadata["schema_version"] == STUDIO_SIMULATION_RUN_SCHEMA_VERSION
    assert metadata["source"] == "ode"
    assert metadata["evidence_classification"] == "simulation"
    assert metadata["sample_count"] == len(response.json()["time"])
    assert metadata["state_variables"] == ["v"]
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["input_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["result_sha256"])
    assert "path" not in metadata


def test_model_simulation_endpoint_returns_run_metadata(client: TestClient) -> None:
    """Named-model simulation responses include source-aware metadata."""

    response = client.post(
        "/api/models/simulate",
        json={"name": "AdExNeuron", "duration": 1.0, "current": 0.0},
    )

    assert response.status_code == 200
    metadata = response.json()["run_metadata"]
    assert metadata["schema_version"] == STUDIO_SIMULATION_RUN_SCHEMA_VERSION
    assert metadata["source"] == "model"
    assert metadata["evidence_classification"] == "simulation"
    assert metadata["sample_count"] == len(response.json()["time"])
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["input_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["result_sha256"])
