# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The fitting routes: a real fit, its replay, and every refusal

"""Fits run through the Studio application exactly as the browser sends them."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.fitting import simulate
from sc_neurocore.neurons.universal_dsl import load_schema
from sc_neurocore.studio.api.fits import (
    MAX_SYNC_FIT_STEPS,
    FitRequest,
    _schema,
    estimated_fit_steps,
)
from sc_neurocore.studio.app import create_app

TRUTH = {"v_rest": -65.0, "tau_m": 10.0, "R": 1.0, "C": 1.0}


def _recording(name: str, level: float, rng: np.random.Generator) -> dict[str, Any]:
    current = [0.0] * 10 + [level] * 70
    trace = simulate(load_schema("lif"), "v", TRUTH, current)
    assert trace is not None
    return {
        "name": name,
        "current": current,
        "observed": [float(value) for value in trace + rng.normal(0.0, 0.2, len(current))],
    }


def _body(**overrides: Any) -> dict[str, Any]:
    rng = np.random.default_rng(11)
    body: dict[str, Any] = {
        "schema": load_schema("lif"),
        "observable": "v",
        "domains": [
            {"name": "v_rest", "low": -80.0, "high": -50.0},
            {"name": "tau_m", "low": 1.0, "high": 50.0, "scale": "log"},
        ],
        "fixed": {"R": 1.0, "C": 1.0},
        "train": [_recording("step +5", 5.0, rng), _recording("step -4", -4.0, rng)],
        "holdout": [_recording("step +8", 8.0, rng)],
        "seed": 2,
        "generations": 12,
        "population": 8,
    }
    body.update(overrides)
    return body


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_a_fit_recovers_the_parameters_and_replays(client: TestClient) -> None:
    response = client.post("/api/fits", json=_body())
    assert response.status_code == 200
    result = response.json()
    assert result["identifiability"]["identifiable"] is True
    for name in ("v_rest", "tau_m"):
        error = result["uncertainty"]["standard_errors"][name]
        assert abs(result["fitted"][name] - TRUTH[name]) <= 4.0 * error, name
    assert result["holdout"][0]["recording"] == "step +8"

    replayed = client.post("/api/fits/replay", json={"result": result})
    assert replayed.status_code == 200
    assert replayed.json()["reproduced"] is True


def test_a_fit_too_large_for_a_request_is_refused_with_its_estimate(client: TestClient) -> None:
    body = _body(generations=500, population=100)
    response = client.post("/api/fits", json=body)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason"] == "fit_too_large"
    assert detail["max_steps"] == MAX_SYNC_FIT_STEPS
    assert detail["estimated_steps"] == estimated_fit_steps(FitRequest.model_validate(body))
    assert detail["estimated_steps"] > MAX_SYNC_FIT_STEPS


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"holdout": []}, "at least one training and one hold-out recording"),
        ({"catalogue_model": "AdExNeuron"}, "not both and not neither"),
        ({"schema": None}, "not both and not neither"),
        ({"schema": None, "catalogue_model": "ATypeKNeuron"}, "has no canonical schema to fit"),
        ({"domains": [{"name": "gain", "low": 0.0, "high": 1.0}]}, "gain is not a parameter"),
    ],
)
def test_an_invalid_fit_is_refused_with_the_reason(
    client: TestClient, overrides: dict[str, Any], message: str
) -> None:
    response = client.post("/api/fits", json=_body(**overrides))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason"] == "invalid_fit"
    assert message in detail["message"]


def test_a_catalogue_model_is_fitted_through_its_canonical_schema() -> None:
    request = FitRequest.model_validate(_body(schema=None, catalogue_model="AdExNeuron"))
    assert _schema(request) == load_schema("adex")


def test_a_result_that_is_not_a_fit_cannot_be_replayed(client: TestClient) -> None:
    response = client.post("/api/fits/replay", json={"result": {"problem": {}}})
    assert response.status_code == 422
    assert response.json()["detail"]["message"].startswith("the result cannot be replayed")


def _exported_problem() -> dict[str, Any]:
    """Return a complete exported fit problem, the part a replay reads first."""
    from sc_neurocore.studio.api.fits import fit_problem

    return fit_problem(FitRequest.model_validate(_body())).to_public_dict()


@pytest.mark.parametrize(
    "result",
    [
        pytest.param({}, id="problem-missing"),
        pytest.param({"problem": 5, "provenance": {}}, id="problem-integer"),
        pytest.param({"problem": "lif", "provenance": {}}, id="problem-string"),
        pytest.param("exported", id="provenance-missing"),
        pytest.param("provenance-list", id="provenance-list"),
    ],
)
def test_a_malformed_replay_names_no_key_or_python_type(client: TestClient, result: Any) -> None:
    """Structural faults are refused with one sentence; the server never errors."""
    from sc_neurocore.studio.api.fits import MALFORMED_DOCUMENT

    if result == "exported":
        result = {"problem": _exported_problem()}
    elif result == "provenance-list":
        result = {"problem": _exported_problem(), "provenance": []}
    response = client.post("/api/fits/replay", json={"result": result})
    assert response.status_code == 422
    assert response.json()["detail"] == {
        "reason": "invalid_fit",
        "message": f"the result cannot be replayed: {MALFORMED_DOCUMENT}",
    }
    for leaked in ("'problem'", "'provenance'", "int", "str", "list", "attribute"):
        assert leaked not in response.text


def test_refusal_message_keeps_authored_reasons_and_hides_structural_faults() -> None:
    """Only explicitly marked laboratory messages are caller-facing."""
    from sc_neurocore.studio.api.fits import MALFORMED_DOCUMENT, refusal_message

    from sc_neurocore.fitting.refusals import LaboratoryRefusal

    assert (
        refusal_message(LaboratoryRefusal("gain is not a parameter")) == "gain is not a parameter"
    )
    for fault in (
        ValueError("gain is not a parameter"),
        KeyError("secret_key"),
        IndexError("list index out of range"),
        TypeError("'int' object is not subscriptable"),
        AttributeError("'int' object has no attribute 'get'"),
    ):
        assert refusal_message(fault) == MALFORMED_DOCUMENT


@pytest.mark.parametrize("route", ["/api/fits/replay", "/api/fits/replay/jobs"])
@pytest.mark.parametrize("fault", ["bound", "current", "generations"])
def test_replay_hides_generated_conversion_and_validation_text(
    client: TestClient, route: str, fault: str
) -> None:
    """Real HTTP replay refuses malformed exported values without echoing conversions."""
    from sc_neurocore.studio.api.fits import MALFORMED_DOCUMENT

    problem = _exported_problem()
    provenance: dict[str, Any] = {"generations": 1, "population": 4}
    if fault == "bound":
        problem["domains"][0]["low"] = "caller-text-xyz"
    elif fault == "current":
        problem["train"][0]["current"] = ["caller-text-xyz"]
    else:
        provenance["generations"] = "caller-text-xyz"
    response = client.post(route, json={"result": {"problem": problem, "provenance": provenance}})
    assert response.status_code == 422
    prefix = "the result cannot be replayed: " if route == "/api/fits/replay" else ""
    assert response.json()["detail"]["message"] == prefix + MALFORMED_DOCUMENT
    for phrase in (
        "caller-text-xyz",
        "could not convert",
        "invalid literal",
        "object has no attribute",
        "is not subscriptable",
        "validation error",
        "errors.pydantic.dev",
    ):
        assert phrase not in response.text
