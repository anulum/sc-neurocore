# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio effective experiment contract (HTTP surface)

"""HTTP contract of the experiment specification, cache and randomness.

``POST /api/simulate`` and ``POST /api/models/simulate`` return the resolved
``experiment`` with its digest, cache by that digest only, never cache a fresh
stochastic trial, refuse an oversized synchronous run with a job pointer, and
reject protocol typos and seeds on deterministic models; the analysis job
route accepts ``simulate`` for what the synchronous routes refuse.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.experiment_spec import EXPERIMENT_SCHEMA_VERSION
from sc_neurocore.studio.platform import StudioRuntimeSettings

ATIF = "AdaptiveThresholdIFNeuron"
MODEL = "/api/models/simulate"
ODE = "/api/simulate"
NOISY = ["dv/dt = -v + I + xi"]


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(create_app(), base_url="http://127.0.0.1") as test_client:
        yield test_client


def _post(client: TestClient, route: str, body: dict[str, Any]) -> Any:
    return client.post(route, json=body)


class TestExperimentBlock:
    def test_model_response_carries_the_experiment_and_manifest_digest(
        self, client: TestClient
    ) -> None:
        response = _post(client, MODEL, {"model_name": ATIF, "duration": 5.0})
        assert response.status_code == 200
        data = response.json()
        experiment = data["experiment"]
        assert experiment["schema_version"] == EXPERIMENT_SCHEMA_VERSION
        assert experiment["model"]["class_name"] == ATIF
        assert experiment["randomness"]["kind"] == "none"
        assert data["cache"]["key"] == experiment["experiment_sha256"]
        assert data["run_metadata"]["experiment_sha256"] == experiment["experiment_sha256"]
        assert data["run_metadata"]["trial"] == "replay"

    def test_gui_defaults_and_implicit_defaults_hit_the_same_entry(
        self, client: TestClient
    ) -> None:
        implicit = _post(client, MODEL, {"model_name": "LapicqueNeuron", "duration": 7.0}).json()
        explicit = _post(
            client,
            MODEL,
            {
                "model_name": "LapicqueNeuron",
                "params": {},
                "duration": 7.0,
                "current": 10.0,
                "protocol": "constant",
                "frequency_hz": 10.0,
                "trial": "replay",
            },
        ).json()
        assert (
            explicit["experiment"]["experiment_sha256"]
            == implicit["experiment"]["experiment_sha256"]
        )
        assert explicit["cache"]["hit"] is True
        assert explicit["raw"] == implicit["raw"]

    def test_changed_sine_frequency_and_dt_are_different_experiments(
        self, client: TestClient
    ) -> None:
        base = {"model_name": ATIF, "duration": 5.0, "protocol": "sine"}
        ten = _post(client, MODEL, {**base, "frequency_hz": 10.0}).json()
        fifty = _post(client, MODEL, {**base, "frequency_hz": 50.0}).json()
        assert ten["experiment"]["experiment_sha256"] != fifty["experiment"]["experiment_sha256"]
        assert ten["raw"]["drive"] != fifty["raw"]["drive"]
        macro = _post(client, MODEL, {"model_name": ATIF, "duration": 5.0, "dt": 0.1}).json()
        micro = _post(client, MODEL, {"model_name": ATIF, "duration": 5.0, "dt": 0.05}).json()
        assert macro["n_steps"] == 50 and micro["n_steps"] == 100
        assert macro["experiment"]["experiment_sha256"] != micro["experiment"]["experiment_sha256"]

    def test_protocol_typo_is_rejected_on_both_routes(self, client: TestClient) -> None:
        for route, body in (
            (MODEL, {"model_name": ATIF, "protocol": "sawtooth"}),
            (ODE, {"equations": ["dv/dt = -v"], "protocol": "sawtooth"}),
        ):
            response = _post(client, route, body)
            assert response.status_code == 422
            assert any(error["type"] == "literal_error" for error in response.json()["detail"])

    def test_unknown_field_is_rejected_on_the_playground_route(self, client: TestClient) -> None:
        response = _post(client, ODE, {"equations": ["dv/dt = -v"], "rng": 1})
        assert response.status_code == 422


class TestRandomness:
    def test_seed_on_deterministic_model_is_experiment_rejected(self, client: TestClient) -> None:
        response = _post(client, MODEL, {"model_name": "AdExNeuron", "seed": 5, "duration": 2.0})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"] == "experiment_rejected"
        assert detail["field"] == "seed"
        assert detail["execution_mode"] == "refused"

    def test_fresh_trials_are_never_cached_and_replay_by_drawn_seed(
        self, client: TestClient
    ) -> None:
        body = {"model_name": "PoissonNeuron", "duration": 100.0, "trial": "fresh"}
        first = _post(client, MODEL, body).json()
        second = _post(client, MODEL, body).json()
        assert first["cache"]["hit"] is False and second["cache"]["hit"] is False
        assert first["experiment"]["randomness"]["seed_source"] == "drawn"
        replay = _post(
            client,
            MODEL,
            {
                "model_name": "PoissonNeuron",
                "duration": 100.0,
                "seed": first["experiment"]["randomness"]["seed"],
            },
        ).json()
        assert replay["spikes"] == first["spikes"]
        assert replay["experiment"]["randomness"]["seed_source"] == "request"

    def test_playground_noise_replays_by_seed_and_is_fresh_on_request(
        self, client: TestClient
    ) -> None:
        seeded = {"equations": NOISY, "duration": 3.0, "seed": 9}
        first = _post(client, ODE, seeded).json()
        assert first["cache"]["hit"] is False
        second = _post(client, ODE, seeded).json()
        assert second["cache"]["hit"] is True
        assert second["raw"]["states"]["v"] == first["raw"]["states"]["v"]
        fresh = _post(client, ODE, {"equations": NOISY, "duration": 3.0, "trial": "fresh"}).json()
        assert fresh["cache"]["hit"] is False
        assert fresh["experiment"]["randomness"]["seed_source"] == "drawn"
        assert fresh["run_metadata"]["trial"] == "fresh"
        deterministic_with_seed = _post(client, ODE, {"equations": ["dv/dt = -v"], "seed": 1})
        assert deterministic_with_seed.status_code == 422
        assert deterministic_with_seed.json()["detail"]["error"] == "experiment_rejected"


class TestOversizedRuns:
    def test_synchronous_route_refuses_with_a_job_pointer(self, client: TestClient) -> None:
        response = _post(client, MODEL, {"model_name": "AdExNeuron", "duration": 1e7})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"] == "experiment_rejected"
        assert detail["execution_mode"] == "job_required"
        assert "analysis=simulate" in detail["recommended_route"]
        assert "not shortened" in detail["reason"]
        playground = _post(client, ODE, {"equations": ["dv/dt = -v"], "duration": 1e6, "dt": 0.1})
        assert playground.status_code == 422
        assert playground.json()["detail"]["execution_mode"] == "job_required"

    def test_simulate_job_runs_the_same_contract_asynchronously(self) -> None:
        with TestClient(create_app(StudioRuntimeSettings()), base_url="http://127.0.0.1") as client:
            submitted = client.post(
                "/api/analysis/jobs",
                json={
                    "analysis": "simulate",
                    "payload": {"model_name": ATIF, "duration": 5.0, "trial": "replay"},
                },
            )
            assert submitted.status_code == 200, submitted.text
            receipt = submitted.json()
            assert receipt["execution_mode"] == "async_job"
            assert receipt["analysis"] == "simulate"
            deadline = time.monotonic() + 20.0
            record: dict[str, Any] = {}
            while time.monotonic() < deadline:
                record = client.get(receipt["status_route"]).json()
                if record.get("status") in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.05)
            assert record.get("status") == "completed", record
            result = record["result"]
            assert result["experiment"]["schema_version"] == EXPERIMENT_SCHEMA_VERSION
            assert (
                result["run_metadata"]["experiment_sha256"]
                == result["experiment"]["experiment_sha256"]
            )
            assert result["state_layout"]["recorded"] == ["v", "theta"]
            rejected = client.post(
                "/api/analysis/jobs",
                json={"analysis": "simulate", "payload": {"model_name": "AdExNeuron", "seed": 1}},
            )
            assert rejected.status_code == 422


class TestOpenApiContract:
    def test_experiment_rejection_is_declared(self, client: TestClient) -> None:
        document = client.get("/openapi.json").json()
        components = document["components"]["schemas"]
        assert (
            components["ExperimentRejectedDetail"]["properties"]["error"]["const"]
            == "experiment_rejected"
        )
        assert components["SimulateRequest"]["additionalProperties"] is False
        assert set(components["SimulateRequest"]["properties"]) >= {
            "frequency_hz",
            "seed",
            "trial",
            "protocol",
        }
        assert set(components["ModelSimulateRequest"]["properties"]) >= {
            "frequency_hz",
            "seed",
            "trial",
        }
        assert "simulate" in components["AnalysisJobRequest"]["properties"]["analysis"]["enum"]
        schema = document["paths"][ODE]["post"]["responses"]["422"]["content"]["application/json"][
            "schema"
        ]
        assert schema["$ref"].endswith("/ModelRunErrorResponse")
