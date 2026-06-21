# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio compile traceability tests

"""Tests for Studio compile source-to-RTL traceability contracts."""

from __future__ import annotations

import hashlib
import json
from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.compile_traceability import (
    STUDIO_COMPILE_TRACEABILITY_SCHEMA_VERSION,
    JsonValue,
    StudioCompileTraceability,
    build_compile_traceability,
)


LIF_COMPILE_REQUEST: dict[str, JsonValue] = {
    "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
    "threshold": "v > -50",
    "reset": "v = -65",
    "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
    "init": {"v": -65.0},
    "module_name": "sc_traceable_neuron",
}


@pytest.fixture
def client() -> TestClient:
    """Return a Studio test client."""

    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_build_compile_traceability_records_source_and_output_hashes() -> None:
    """Compile traceability records source payload and RTL digests without paths."""

    verilog = "module sc_traceable_neuron; endmodule\n"
    traceability = build_compile_traceability(
        equations=["dv/dt = -v / tau"],
        threshold="v > 1",
        reset="v = 0",
        params={"tau": 10.0},
        init={"v": 0.0},
        module_name="sc_traceable_neuron",
        verilog=verilog,
    ).to_public_dict()

    assert traceability["schema_version"] == STUDIO_COMPILE_TRACEABILITY_SCHEMA_VERSION
    assert traceability["evidence_classification"] == "compile"
    assert traceability["status"] == "completed"
    assert traceability["source"] == "ode"
    assert traceability["input_sha256"] == _sha256_json(
        cast(dict[str, JsonValue], traceability["source_payload"])
    )
    output = cast(dict[str, JsonValue], traceability["output"])
    assert output["module_name"] == "sc_traceable_neuron"
    assert output["rtl_chars"] == len(verilog)
    assert output["rtl_sha256"] == hashlib.sha256(verilog.encode("utf-8")).hexdigest()
    assert "path" not in json.dumps(traceability, sort_keys=True).lower()


def test_build_compile_traceability_rejects_missing_equations() -> None:
    """Compile traceability fails closed when the source equations are absent."""

    with pytest.raises(ValueError, match="At least one equation"):
        build_compile_traceability(
            equations=[],
            threshold=None,
            reset=None,
            params=None,
            init=None,
            module_name="sc_traceable_neuron",
            verilog="module sc_traceable_neuron; endmodule\n",
        )


def test_compile_traceability_rejects_unknown_evidence_classification() -> None:
    """Compile traceability uses the shared Studio evidence-class contract."""

    traceability = StudioCompileTraceability(
        equations=("dv/dt = -v / tau",),
        threshold="v > 1",
        reset="v = 0",
        params={"tau": 10.0},
        init={"v": 0.0},
        module_name="sc_traceable_neuron",
        verilog="module sc_traceable_neuron; endmodule\n",
        evidence_classification="screenshots",  # type: ignore[arg-type]  # Invalid by design.
    )

    with pytest.raises(ValueError, match="classification"):
        traceability.to_public_dict()


def test_compile_traceability_rejects_unknown_status() -> None:
    """Compile traceability uses the shared terminal-status contract."""

    traceability = StudioCompileTraceability(
        equations=("dv/dt = -v / tau",),
        threshold="v > 1",
        reset="v = 0",
        params={"tau": 10.0},
        init={"v": 0.0},
        module_name="sc_traceable_neuron",
        verilog="module sc_traceable_neuron; endmodule\n",
        status="running",  # type: ignore[arg-type]  # Invalid by design.
    )

    with pytest.raises(ValueError, match="status"):
        traceability.to_public_dict()


def test_compile_route_returns_traceability_manifest(client: TestClient) -> None:
    """The public compile route returns source-to-RTL traceability."""

    response = client.post("/api/compile", json=LIF_COMPILE_REQUEST)

    assert response.status_code == 200
    payload = response.json()
    traceability = cast(dict[str, JsonValue], payload["compile_traceability"])
    assert traceability["schema_version"] == STUDIO_COMPILE_TRACEABILITY_SCHEMA_VERSION
    assert traceability["evidence_classification"] == "compile"
    assert traceability["status"] == "completed"
    assert traceability["source"] == "ode"
    assert traceability["input_sha256"] == _sha256_json(
        cast(dict[str, JsonValue], traceability["source_payload"])
    )
    output = cast(dict[str, JsonValue], traceability["output"])
    assert output["module_name"] == payload["module_name"] == "sc_traceable_neuron"
    assert output["rtl_chars"] == payload["chars"]
    assert output["rtl_sha256"] == hashlib.sha256(payload["verilog"].encode("utf-8")).hexdigest()


def test_direct_sv_route_returns_traceability_manifest(client: TestClient) -> None:
    """The Compiler Inspector direct-SV route returns the same manifest contract."""

    response = client.post("/api/ir/emit-sv-direct", json=LIF_COMPILE_REQUEST)

    assert response.status_code == 200
    payload = response.json()
    traceability = cast(dict[str, JsonValue], payload["compile_traceability"])
    output = cast(dict[str, JsonValue], traceability["output"])
    assert traceability["schema_version"] == STUDIO_COMPILE_TRACEABILITY_SCHEMA_VERSION
    assert traceability["status"] == "completed"
    assert output["module_name"] == payload["module_name"] == "sc_ode_neuron"
    assert output["rtl_sha256"] == hashlib.sha256(payload["verilog"].encode("utf-8")).hexdigest()


def test_compile_route_rejects_empty_equation_list(client: TestClient) -> None:
    """The compile request model rejects empty source equation lists."""

    response = client.post("/api/compile", json={**LIF_COMPILE_REQUEST, "equations": []})

    assert response.status_code == 422


def _sha256_json(payload: dict[str, JsonValue]) -> str:
    """Return a stable SHA-256 digest over canonical JSON."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
