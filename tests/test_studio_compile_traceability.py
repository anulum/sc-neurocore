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
import shutil
from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.compile_traceability import (
    STUDIO_COMPILE_TRACEABILITY_SCHEMA_VERSION,
    JsonValue,
    StudioCompileTraceability,
    build_compile_traceability,
    build_model_compile_traceability,
)
from sc_neurocore.studio.platform import StudioRuntimeSettings


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


def test_build_model_compile_traceability_records_selected_configuration() -> None:
    """Model traceability hashes the canonical model, schema, solver, and precision."""

    traceability = build_model_compile_traceability(
        model_name="SCLapicqueLIFNeuron",
        schema_name="sc_lapicque_lif",
        schema_sha256="a" * 64,
        params={"tau": 20.0},
        dt=1.0,
        integrator="exp_euler",
        q_format="Q8.8",
        module_name="sc_lapicque",
        verilog="module sc_lapicque; endmodule\n",
    ).to_public_dict()

    assert traceability["source"] == "model"
    assert traceability["source_payload"] == {
        "dt": 1.0,
        "integrator": "exp_euler",
        "model_name": "SCLapicqueLIFNeuron",
        "params": {"tau": 20.0},
        "q_format": "Q8.8",
        "schema_name": "sc_lapicque_lif",
        "schema_sha256": "a" * 64,
    }
    assert traceability["input_sha256"] == _sha256_json(
        cast(dict[str, JsonValue], traceability["source_payload"])
    )


def test_build_model_compile_traceability_requires_schema_identity() -> None:
    with pytest.raises(ValueError, match="schema digest"):
        build_model_compile_traceability(
            model_name="SCLapicqueLIFNeuron",
            schema_name="sc_lapicque_lif",
            schema_sha256="short",
            params=None,
            dt=1.0,
            integrator="exp_euler",
            q_format="Q8.8",
            module_name="sc_lapicque",
            verilog="module sc_lapicque; endmodule\n",
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


def test_model_compile_route_emits_real_schema_backed_rtl(client: TestClient) -> None:
    """The public model route compiles the selected schema and returns its configuration."""

    response = client.post(
        "/api/models/compile",
        json={
            "model_name": "SCLapicqueLIFNeuron",
            "params": {"tau": 15.0},
            "dt": 1.0,
            "integrator": "exp_euler",
            "q_format": "Q8.8",
            "module_name": "sc_studio_lapicque",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "module sc_studio_lapicque" in payload["verilog"]
    assert payload["compile_configuration"] == {
        "dt": 1.0,
        "integrator": "exp_euler",
        "model_name": "SCLapicqueLIFNeuron",
        "q_format": "Q8.8",
        "schema_name": "sc_lapicque_lif",
        "schema_sha256": payload["compile_traceability"]["source_payload"]["schema_sha256"],
    }
    assert len(payload["compile_configuration"]["schema_sha256"]) == 64
    assert payload["compile_traceability"]["source"] == "model"


def test_model_compile_route_is_authenticated_when_policy_is_enforced() -> None:
    """The new compute route participates in the production fail-closed policy registry."""

    protected_client = TestClient(
        create_app(StudioRuntimeSettings(enforce_route_policies=True)),
        base_url="http://127.0.0.1",
    )
    response = protected_client.post(
        "/api/models/compile",
        json={"model_name": "SCLapicqueLIFNeuron", "q_format": "Q8.8"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "missing_principal"


@pytest.mark.skipif(
    not all(shutil.which(tool) is not None for tool in ("gcc", "iverilog", "vvp")),
    reason="GCC and Icarus Verilog are required",
)
def test_model_cosim_route_returns_real_bit_exact_selected_configuration(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/models/cosim",
        json={
            "model_name": "AdaptiveThresholdIFNeuron",
            "integrator": "map",
            "q_format": "Q8.8",
            "current": 10.0,
            "n_steps": 12,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["schema_version"] == "studio.cosim-parity.v1"
    assert payload["bit_exact"] is True
    assert payload["configuration"]["model_name"] == "AdaptiveThresholdIFNeuron"
    assert payload["rtl"]["trace_sha256"] == payload["reference"]["trace_sha256"]
    assert "path" not in json.dumps(payload, sort_keys=True).lower()


def test_model_cosim_route_is_authenticated_when_policy_is_enforced() -> None:
    protected_client = TestClient(
        create_app(StudioRuntimeSettings(enforce_route_policies=True)),
        base_url="http://127.0.0.1",
    )
    response = protected_client.post(
        "/api/models/cosim",
        json={"model_name": "AdaptiveThresholdIFNeuron", "current": 1.0, "n_steps": 4},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "missing_principal"


@pytest.mark.parametrize(
    "payload",
    [
        {"model_name": "MissingNeuron", "q_format": "Q8.8"},
        {"model_name": "SCLapicqueLIFNeuron", "integrator": "rk4", "q_format": "Q8.8"},
        {"model_name": "SCLapicqueLIFNeuron", "q_format": "Q1.0"},
    ],
)
def test_model_compile_route_fails_closed_for_invalid_configuration(
    client: TestClient, payload: dict[str, JsonValue]
) -> None:
    response = client.post("/api/models/compile", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "studio_job_failed"


def _sha256_json(payload: dict[str, JsonValue]) -> str:
    """Return a stable SHA-256 digest over canonical JSON."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
