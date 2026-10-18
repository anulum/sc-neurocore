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
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.network_graph import create_population


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Serve the production routers under the supported loopback profile."""
    with TestClient(
        create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    ) as http:
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
@pytest.mark.parametrize("seed", [None, 9007199254740993, 9223372036854775807])
def test_existing_model_rule_arrives_verbatim(
    client: TestClient, route: str, seed: int | None
) -> None:
    """Deliberate parameter validation retains the original public rule."""
    model = "AdaptiveThresholdIFNeuron" if seed is None else "PoissonNeuron"
    response = client.post(
        route, json={"model_name": model, "params": {"dt": 0.1} if seed is None else {"seed": seed}}
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    if seed is None:
        assert detail["field"] == "params.dt"
        assert (
            detail["reason"] == "the timestep is set through the dt field, not a parameter override"
        )
    else:
        if route == "/api/export/svg":
            assert detail["field"] == "constructor"
            assert detail["reason"] == "model constructor rejected the supplied parameters"
        else:
            assert detail["field"] == "seed"
            assert (
                detail["reason"] == f"PoissonNeuron accepts a seed in [0, 65535]; received {seed}"
            )


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


@pytest.mark.parametrize("dt", [0.25, 1.0, 2.0])
def test_profile_timebase_matches_direct_poisson_dynamics(client: TestClient, dt: float) -> None:
    """Bind the declared bin width and check every event/state and observation time."""
    from sc_neurocore.neurons.models.poisson import PoissonNeuron

    response = client.post(
        "/api/models/simulate",
        json={"model_name": "PoissonNeuron", "dt": dt, "duration": 20, "current": 100, "seed": 123},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    reference = PoissonNeuron(dt_ms=dt, seed=123)
    states: list[int] = []
    spikes: list[int] = []
    for index in range(result["n_steps"]):
        if reference.step(100):
            spikes.append(index)
        states.append(reference.rng_state)
    assert result["dt"] == dt
    assert result["n_steps"] == int(20 / dt)
    assert result["time"] == [(i + 1) * dt for i in range(result["n_steps"])]
    assert result["raw"]["states"]["rng_state"] == states
    assert result["raw"]["spike_indices"] == spikes
    assert result["effective_inputs"]["parameters"]["dt_ms"] == dt
    assert result["final_state"]["rng_state"] == reference.rng_state


@pytest.mark.parametrize("run_dt", [None, 0.5])
def test_legacy_timebase_override_resolves_one_clock(
    client: TestClient, run_dt: float | None
) -> None:
    """A legacy alias alone or matching the run timestep defines one effective clock."""
    body: dict[str, object] = {
        "model_name": "PoissonNeuron",
        "params": {"dt_ms": 0.5},
        "duration": 20,
        "seed": 123,
    }
    if run_dt is not None:
        body["dt"] = run_dt
    response = client.post("/api/models/simulate", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["dt"] == 0.5
    assert response.json()["n_steps"] == 40


def test_conflicting_constructor_timebase_is_refused(client: TestClient) -> None:
    """Never run a Poisson bin width on a contradictory observation clock."""
    response = client.post(
        "/api/models/simulate",
        json={"model_name": "PoissonNeuron", "params": {"dt_ms": 0.5}, "dt": 1.0},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == {
        "error": "invalid_model_input",
        "model": "PoissonNeuron",
        "field": "params.dt_ms",
        "reason": "timebase parameter contradicts the run dt field",
    }


def test_non_millisecond_profile_is_refused_before_execution(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real profile resource with incompatible units cannot mislabel the run clock."""
    from sc_neurocore.neurons import universal_dsl

    source = (universal_dsl._SCHEMA_DIR / "poisson.toml").read_text()
    changed = source.replace('time_unit = "ms"', 'time_unit = "s"').replace(
        'dt_ms = "ms"', 'dt_ms = "s"'
    )
    assert changed != source
    (tmp_path / "poisson.toml").write_text(changed)
    monkeypatch.setattr(universal_dsl, "_SCHEMA_DIR", tmp_path)
    response = client.post("/api/models/simulate", json={"model_name": "PoissonNeuron", "dt": 1.0})
    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "dt"
    assert response.json()["detail"]["reason"] == (
        "model does not declare one constructor timebase in milliseconds"
    )


def test_missing_profile_refuses_an_unverifiable_clock(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing canonical clock resource is refused rather than defaulted."""
    from sc_neurocore.neurons import universal_dsl

    monkeypatch.setattr(universal_dsl, "_SCHEMA_DIR", tmp_path)
    response = client.post(
        "/api/models/simulate",
        json={"model_name": "IntegerQIFNeuron", "dt": 1.0, "duration": 10, "current": 10},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["field"] == "dt"
    assert response.json()["detail"]["reason"] == "model timestep profile is unavailable or invalid"
