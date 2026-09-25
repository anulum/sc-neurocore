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
