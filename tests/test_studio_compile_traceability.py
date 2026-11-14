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
from pathlib import Path
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
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.model_compile_configuration import resolve_model_compile_configuration
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
    return TestClient(create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"})


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
    assert traceability["input_sha256"] == _sha256_json(traceability["source_payload"])


def test_build_model_compile_traceability_requires_schema_identity() -> None:
    """Require canonical model schema identity before constructing compile traceability."""
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


def _studio_model_request(model_name: str, seed: int | None) -> dict[str, object]:
    """Build a model compile request the way the Studio's 'Generate RTL' does.

    Parameters
    ----------
    model_name : str
        A catalogue model with an RTL path.
    seed : int or None
        A seed to send in place of the catalogue's; None sends the defaults alone.

    Returns
    -------
    dict[str, object]
        The model's step, default integrator and Q-format, and catalogue defaults.
    """
    detail = get_model_detail(model_name)
    assert detail is not None
    choices = detail["compile_configuration"]
    params = {parameter["name"]: parameter["default"] for parameter in detail["params"]}
    if seed is not None:
        params["seed"] = float(seed)
    return {
        "model_name": model_name,
        "dt": detail["dt"],
        "integrator": choices["default_integrator"],
        "q_format": choices["default_q_format"],
        "params": params,
    }


def _stated_source_payload(model_name: str, seed: int | None) -> dict[str, JsonValue]:
    """Return the traceability input the resolved Studio request states, field by field.

    Parameters
    ----------
    model_name : str
        A catalogue model with an RTL path.
    seed : int or None
        A seed to send in place of the catalogue's; None sends the defaults alone.

    Returns
    -------
    dict[str, JsonValue]
        The configuration's step, integrator, model, format, schema and translated
        parameters, with its seed when it has one.
    """
    configuration = resolve_model_compile_configuration(_studio_model_request(model_name, seed))
    assert configuration.rng_seed == seed
    stated = configuration.to_public_dict()
    payload: dict[str, JsonValue] = {
        key: cast(JsonValue, stated[key])
        for key in ("dt", "integrator", "model_name", "q_format", "schema_name", "schema_sha256")
    }
    payload["params"] = cast(dict[str, JsonValue], dict(configuration.params))
    if seed is not None:
        payload["rng_seed"] = seed
    return payload


#: Two seeded models with a seed other than the catalogue's, and a deterministic one.
SEEDED_AND_DETERMINISTIC = [
    ("EscapeRateNeuron", 1234),
    ("PoissonNeuron", 977),
    ("LapicqueNeuron", None),
]


@pytest.mark.parametrize(("model_name", "seed"), SEEDED_AND_DETERMINISTIC)
def test_a_resolved_model_compile_states_its_configuration_in_its_traceability(
    model_name: str, seed: int | None
) -> None:
    """The traceability a resolved compile builds states its configuration, seed included.

    The model compile worker and the hardware path both record this traceability,
    so neither can leave out a field the configuration states. A seeded model's
    request carries its seed; a deterministic model's input has none.

    Parameters
    ----------
    model_name : str
        A catalogue model with an RTL path.
    seed : int or None
        The seed sent with a seeded model; None for a deterministic one.
    """
    configuration = resolve_model_compile_configuration(_studio_model_request(model_name, seed))
    verilog = configuration.to_verilog()

    traceability = configuration.to_traceability(verilog).to_public_dict()

    assert traceability["source"] == "model"
    assert traceability["source_payload"] == _stated_source_payload(model_name, seed)
    output = cast(dict[str, JsonValue], traceability["output"])
    assert output["module_name"] == configuration.module_name
    assert output["rtl_sha256"] == hashlib.sha256(verilog.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(("model_name", "seed"), SEEDED_AND_DETERMINISTIC)
def test_the_model_compile_route_binds_its_traceability_to_its_request_and_rtl(
    tmp_path: Path, model_name: str, seed: int | None
) -> None:
    """The compile worker's traceability states the request it resolved and the RTL it returns.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    model_name : str
        A catalogue model with an RTL path.
    seed : int or None
        The seed sent with a seeded model; None for a deterministic one.
    """
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"), audit_log_path=str(tmp_path / "audit.jsonl")
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as studio:
        response = studio.post(
            "/api/models/compile",
            json=_studio_model_request(model_name, seed),
            headers={"Origin": "http://127.0.0.1:5173"},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    traceability = cast(dict[str, JsonValue], payload["compile_traceability"])
    assert traceability["source_payload"] == _stated_source_payload(model_name, seed)
    assert payload["compile_configuration"]["rng_seed"] == seed
    output = cast(dict[str, JsonValue], traceability["output"])
    assert output["module_name"] == payload["module_name"]
    assert output["rtl_sha256"] == hashlib.sha256(payload["verilog"].encode("utf-8")).hexdigest()


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
    assert output["module_name"] == payload["module_name"] == "sc_traceable_neuron"
    assert output["rtl_sha256"] == hashlib.sha256(payload["verilog"].encode("utf-8")).hexdigest()


@pytest.mark.parametrize("route", ["/api/compile", "/api/ir/emit-sv-direct"])
def test_compile_routes_preserve_coupled_equations_and_init(client: TestClient, route: str) -> None:
    """Preserve coupled equations and initial state in the generated RTL."""
    from sc_neurocore.compiler.equation_compiler import equation_to_fpga

    request = {
        "equations": ["dv/dt = -v + w + I", "dw/dt = -w"],
        "init": {"v": 2.0, "w": 3.0},
        "module_name": "sc_coupled",
    }
    response = client.post(route, json=request)
    assert response.status_code == 200, response.text
    _, expected = equation_to_fpga(
        "dv/dt = -v + w + I", "dw/dt = -w", init={"v": 2.0, "w": 3.0}, module_name="sc_coupled"
    )
    payload = response.json()
    assert payload["verilog"] == expected
    source = payload["compile_traceability"]["source_payload"]
    assert source["equations"] == request["equations"]
    assert source["init"] == request["init"]


@pytest.mark.parametrize("route", ["/api/compile", "/api/ir/emit-sv-direct"])
@pytest.mark.parametrize("field", ["dt", "duration", "current", "unknown_option"])
def test_compile_routes_reject_unimplemented_options(
    client: TestClient, route: str, field: str
) -> None:
    """Refuse compiler options which the selected backend cannot implement."""
    response = client.post(route, json={**LIF_COMPILE_REQUEST, field: 1.0})
    assert response.status_code == 422


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
    configuration = dict(payload["compile_configuration"])
    profile = configuration.pop("profile")
    contract = configuration.pop("numeric_contract")
    assert (contract["q_format"], contract["representable"]) == ("Q8.8", True)
    # The bit-true C kernel mirrors exponential Euler; the contract states its update.
    assert contract["bit_true_mirror"]["available"] is True
    assert "exprel" in contract["bit_true_mirror"]["arithmetic"]["state_update"]
    assert configuration == {
        "dt": 1.0,
        "integrator": "exp_euler",
        "model_name": "SCLapicqueLIFNeuron",
        "q_format": "Q8.8",
        # A deterministic model's RTL starts no LFSR, so it records no seed.
        "rng_seed": None,
        "schema_name": "sc_lapicque_lif",
        "schema_sha256": payload["compile_traceability"]["source_payload"]["schema_sha256"],
    }
    assert profile["contract"] == "sc-neurocore.model-profile.v1"
    assert profile["method"] == profile["declared_method"] == "exp_euler"
    assert profile["derived"] is False
    assert len(payload["compile_configuration"]["schema_sha256"]) == 64
    assert payload["compile_traceability"]["source"] == "model"


def test_model_compile_route_is_authenticated_when_policy_is_enforced() -> None:
    """The new compute route participates in the production fail-closed policy registry."""
    protected_client = TestClient(
        create_app(StudioRuntimeSettings(enforce_route_policies=True)),
        base_url="http://127.0.0.1",
        headers={"sec-fetch-site": "none"},
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
    """Report real co-simulation parity for the exact selected model configuration."""
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
    """Require a principal before running the protected co-simulation route."""
    protected_client = TestClient(
        create_app(StudioRuntimeSettings(enforce_route_policies=True)),
        base_url="http://127.0.0.1",
        headers={"sec-fetch-site": "none"},
    )
    response = protected_client.post(
        "/api/models/cosim",
        json={"model_name": "AdaptiveThresholdIFNeuron", "current": 1.0, "n_steps": 4},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "missing_principal"


@pytest.mark.parametrize(
    "payload, reason",
    [
        (
            {"model_name": "MissingNeuron", "q_format": "Q8.8"},
            "Unknown Studio model 'MissingNeuron'.",
        ),
        (
            {"model_name": "SCLapicqueLIFNeuron", "integrator": "rk4", "q_format": "Q8.8"},
            "Integrator 'rk4' is not declared for 'SCLapicqueLIFNeuron'.",
        ),
        ({"model_name": "SCLapicqueLIFNeuron", "q_format": "Q1.0"}, None),
    ],
)
def test_model_compile_route_fails_closed_for_invalid_configuration(
    client: TestClient, payload: dict[str, JsonValue], reason: str | None
) -> None:
    """Refuse an unknown model, an undeclared integrator and an unoffered format before any job runs.

    Each is the caller's input, so each is answered 422. A model configuration
    the compiler refuses carries the compiler's own reason; a format is refused
    by the request schema, with its own detail.

    Parameters
    ----------
    client : TestClient
        Real application HTTP client.
    payload : dict of str to JsonValue
        Unsupported format or model configuration to submit.
    reason : str or None
        The compiler's reason for a refused model configuration, or None for a
        format the request schema refuses.
    """
    response = client.post("/api/models/compile", json=payload)

    assert response.status_code == 422
    if reason is not None:
        assert response.json()["detail"] == reason


def _sha256_json(payload: dict[str, JsonValue]) -> str:
    """Return a stable SHA-256 digest over canonical JSON."""
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
