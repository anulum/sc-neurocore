# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored model-run refusals across HTTP consumers

"""Exercise model-run diagnostic producers through their real Studio routes."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.network_graph import create_population
from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Serve the production routers under the supported loopback profile."""
    with TestClient(create_app(), base_url="http://127.0.0.1") as http:
        yield http


@pytest.mark.parametrize("route", ["/api/models/simulate", "/api/export/svg"])
def test_constructor_failure_is_an_authored_field_refusal(client: TestClient, route: str) -> None:
    """An actual rejected constructor keeps its field without exposing its exception."""
    response = client.post(
        route,
        json={"model_name": "AdaptiveThresholdIFNeuron", "params": {"theta_rest": -70.0}},
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "error": "invalid_model_input",
            "model": "AdaptiveThresholdIFNeuron",
            "field": "constructor",
            "reason": "model constructor rejected the supplied parameters",
        }
    }


@pytest.mark.parametrize("route", ["/api/models/simulate", "/api/export/svg"])
def test_catalogue_overflow_keeps_step_metadata_without_interpreter_text(
    client: TestClient, route: str
) -> None:
    """A real Hodgkin–Huxley overflow is refused without yielding a partial trace."""
    response = client.post(
        route,
        json={
            "model_name": "HodgkinHuxleyNeuron",
            "protocol": "ramp",
            "current": 1e300,
            "duration": 5.0,
            "dt": 0.01,
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "model_simulation_failed"
    assert detail["model"] == "HodgkinHuxleyNeuron"
    assert detail["backend"] == "python"
    assert detail["step"] > 0
    assert detail["time_ms"] == pytest.approx(detail["step"] * 1.0)
    assert detail["diagnostic"] == "model step could not produce a finite result"
    assert "OverflowError" not in response.text
    assert "time" not in response.json()


@pytest.mark.parametrize(
    "route", ["/api/simulate", "/api/export/replay-pack", "/api/export/replay-notebook"]
)
def test_equation_overflow_is_refused_before_export(client: TestClient, route: str) -> None:
    """The same real equation failure is refused by simulation and replay exports."""
    body: dict[str, object] = {
        "equations": ["dv/dt = exp(v)"],
        "init": {"v": 1000.0},
        "duration": 1.0,
    }
    if route != "/api/simulate":
        body["mode"] = "ode"
    response = client.post(route, json=body)
    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "error": "model_simulation_failed",
            "model": "ode",
            "backend": "python",
            "step": 0,
            "time_ms": 0.0,
            "diagnostic": "equation step could not produce a finite result",
        }
    }


@pytest.mark.parametrize("route", ["/api/graph/simulate", "/api/graph/notebook"])
def test_graph_execution_fault_is_an_authored_refusal(client: TestClient, route: str) -> None:
    """Actual finite AdEx inputs overflow during graph execution, never in validation."""
    population = create_population(
        model="AdExNeuron",
        count=1,
        params={"c_m": 1e-300},
        drive={"kind": "constant", "current": 1e300},
    )
    population["id"] = "a"
    response = client.post(
        route, json={"populations": [population], "projections": [], "duration": 1.0, "dt": 0.1}
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "error": "graph_execution_failed",
            "reason": "graph execution could not produce a finite result",
        }
    }


@pytest.mark.parametrize("route", ["/api/models/simulate", "/api/export/svg"])
def test_existing_model_rule_arrives_verbatim(client: TestClient, route: str) -> None:
    """Deliberate parameter validation retains the original public rule."""
    response = client.post(
        route, json={"model_name": "AdaptiveThresholdIFNeuron", "params": {"dt": 0.1}}
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["field"] == "params.dt"
    assert detail["reason"] == "the timestep is set through the dt field, not a parameter override"


def test_invalid_model_signature_metadata_has_an_authored_http_reason(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse real introspection failure caused by corrupt metadata on a catalogue step."""
    monkeypatch.setattr(
        AdaptiveThresholdIFNeuron.step, "__signature__", "caller-text-xyz", raising=False
    )
    response = client.post(
        "/api/models/simulate", json={"model_name": "AdaptiveThresholdIFNeuron", "duration": 1.0}
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "error": "invalid_model_input",
            "model": "AdaptiveThresholdIFNeuron",
            "field": "step",
            "reason": "model step signature is unavailable",
        }
    }
    assert "caller-text-xyz" not in response.text
