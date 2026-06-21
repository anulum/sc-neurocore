# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio analysis manifest tests

"""Tests for Studio analysis result manifest contracts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.analysis_manifest import (
    STUDIO_ANALYSIS_RESULT_SCHEMA_VERSION,
    JsonValue,
    attach_analysis_result_manifest,
    build_analysis_result_manifest,
    infer_analysis_source,
)
from sc_neurocore.studio.app import create_app


LIF_REQUEST: dict[str, JsonValue] = {
    "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
    "threshold": "v > -50",
    "reset": "v = -65",
    "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
    "init": {"v": -65.0},
    "dt": 0.1,
    "duration": 20.0,
    "current": 30.0,
}


@pytest.fixture
def client() -> TestClient:
    """Return a Studio test client."""

    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_build_analysis_result_manifest_returns_path_free_hashes() -> None:
    """Analysis manifests record stable request and result digests."""

    request = {"equations": ["dv/dt = -v / tau"], "params": {"tau": 10.0}}
    result = {"currents": [0.0, 1.0], "rates": [0.0, 25.0]}
    manifest = build_analysis_result_manifest(
        analysis_type="fi_curve",
        source="ode",
        request_payload=request,
        result_payload=result,
    ).to_public_dict()

    assert manifest == {
        "analysis_type": "fi_curve",
        "evidence_classification": "analysis",
        "input_sha256": _sha256_json(request),
        "output_keys": ["currents", "rates"],
        "result_sha256": _sha256_json(result),
        "schema_version": STUDIO_ANALYSIS_RESULT_SCHEMA_VERSION,
        "source": "ode",
    }
    assert "path" not in json.dumps(manifest, sort_keys=True).lower()


def test_attach_analysis_result_manifest_ignores_existing_metadata_for_hash() -> None:
    """Attached manifests hash the analysis payload without stale metadata."""

    request = {"model_name": "LIFNeuron"}
    result: dict[str, Any] = {
        "analysis_metadata": {"stale": True},
        "frequencies_hz": [1.0],
        "rates": [2.0],
    }

    attached = attach_analysis_result_manifest(
        analysis_type="frequency_response",
        source="model",
        request_payload=request,
        result_payload=result,
    )

    metadata = cast(dict[str, JsonValue], attached["analysis_metadata"])
    assert metadata["source"] == "model"
    assert metadata["result_sha256"] == _sha256_json({"frequencies_hz": [1.0], "rates": [2.0]})
    assert metadata["output_keys"] == ["frequencies_hz", "rates"]


@pytest.mark.parametrize(
    ("payload", "source"),
    [
        ({"config_a": {}, "config_b": {}}, "mixed"),
        ({"model_name": "LIFNeuron"}, "model"),
        ({"equations": ["dv/dt = -v"]}, "ode"),
        ({}, "unknown"),
    ],
)
def test_infer_analysis_source(payload: dict[str, Any], source: str) -> None:
    """Analysis source inference follows Studio request conventions."""

    assert infer_analysis_source(payload) == source


def test_build_analysis_result_manifest_rejects_non_portable_numbers() -> None:
    """Analysis manifests fail closed when payloads are not portable JSON."""

    with pytest.raises(ValueError, match="portable JSON"):
        build_analysis_result_manifest(
            analysis_type="fi_curve",
            source="ode",
            request_payload={"bad": float("nan")},
            result_payload={"rates": [1.0]},
        )


@pytest.mark.parametrize(
    ("path", "request_payload", "analysis_type", "source"),
    [
        (
            "/api/fi-curve",
            {**LIF_REQUEST, "i_min": 0.0, "i_max": 20.0, "i_steps": 3},
            "fi_curve",
            "ode",
        ),
        (
            "/api/bifurcation",
            {
                **LIF_REQUEST,
                "sweep_param": "tau_m",
                "sweep_min": 8.0,
                "sweep_max": 12.0,
                "sweep_steps": 5,
            },
            "bifurcation",
            "ode",
        ),
        ("/api/sensitivity", LIF_REQUEST, "sensitivity", "ode"),
        ("/api/precision", LIF_REQUEST, "precision", "ode"),
        (
            "/api/freq-response",
            {**LIF_REQUEST, "freq_min": 1.0, "freq_max": 5.0, "n_freqs": 3, "amplitude": 20.0},
            "frequency_response",
            "ode",
        ),
        (
            "/api/heatmap",
            {
                **LIF_REQUEST,
                "param_x": "tau_m",
                "x_min": 8.0,
                "x_max": 12.0,
                "x_steps": 3,
                "param_y": "C",
                "y_min": 0.8,
                "y_max": 1.2,
                "y_steps": 3,
            },
            "heatmap",
            "ode",
        ),
        (
            "/api/nullclines",
            {
                "equations": ["dv/dt = w", "dw/dt = -v"],
                "params": {},
                "var_names": ["v", "w"],
                "ranges": {"v": [-1.0, 1.0], "w": [-1.0, 1.0]},
                "grid_size": 20,
            },
            "nullclines",
            "ode",
        ),
    ],
)
def test_analysis_routes_return_metadata(
    client: TestClient,
    path: str,
    request_payload: dict[str, Any],
    analysis_type: str,
    source: str,
) -> None:
    """Analysis endpoints return the shared metadata contract."""

    response = client.post(path, json=request_payload)

    assert response.status_code == 200
    payload = response.json()
    metadata = cast(dict[str, JsonValue], payload["analysis_metadata"])
    assert metadata["schema_version"] == STUDIO_ANALYSIS_RESULT_SCHEMA_VERSION
    assert metadata["analysis_type"] == analysis_type
    assert metadata["source"] == source
    assert metadata["evidence_classification"] == "analysis"
    assert isinstance(metadata["input_sha256"], str)
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["input_sha256"])
    result_without_metadata = {
        key: value for key, value in payload.items() if key != "analysis_metadata"
    }
    assert metadata["output_keys"] == sorted(result_without_metadata)
    assert metadata["result_sha256"] == _sha256_json(result_without_metadata)


def test_compare_route_returns_mixed_analysis_metadata(client: TestClient) -> None:
    """Comparison analysis records the mixed-source metadata contract."""

    response = client.post(
        "/api/compare",
        json={"config_a": LIF_REQUEST, "config_b": {**LIF_REQUEST, "current": 10.0}},
    )

    assert response.status_code == 200
    metadata = response.json()["analysis_metadata"]
    assert metadata["analysis_type"] == "compare"
    assert metadata["source"] == "mixed"
    assert metadata["output_keys"] == ["a", "b"]


def _sha256_json(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 digest over canonical JSON."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
