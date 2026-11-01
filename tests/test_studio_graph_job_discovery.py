# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public graph job discovery and documented request acceptance

"""Bind the real graph routes, capability discovery and documented worker example."""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
from pathlib import Path
import time

from httpx import Response as HttpxResponse
from httpx2 import Response as Httpx2Response
import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.policy import (
    RouteVisibility,
    build_default_studio_route_policy_registry,
)
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings

ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROUTES = (
    ("POST", "/api/graph/jobs/validate", "studio.graph.job_validate"),
    ("POST", "/api/graph/jobs", "studio.graph.job_submit"),
    ("GET", "/api/graph/jobs", "studio.graph.job_history"),
    ("GET", "/api/graph/jobs/{job_id}", "studio.graph.job_status"),
    ("POST", "/api/graph/jobs/{job_id}/cancel", "studio.graph.job_cancel"),
    ("GET", "/api/graph/jobs/{job_id}/result", "studio.graph.job_result"),
)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Start the actual API and process authority with exclusively owned durable roots."""
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_PROJECT_ROOT", str(tmp_path / "projects"))
    application = create_app(
        StudioRuntimeSettings(
            job_root_path=str(tmp_path / "jobs"),
            audit_log_path=str(tmp_path / "audit.jsonl"),
            benchmark_databank_path=str(tmp_path / "databank.jsonl"),
        )
    )
    with TestClient(
        application, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    ) as session:
        yield session


def _terminal_response(client: TestClient, route: str) -> HttpxResponse | Httpx2Response:
    """Observe the real supervised process under one finite ninety-second deadline."""
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        response = client.get(route)
        assert response.status_code == 200, response.text
        if response.json()["status"] not in {"pending", "running", "cancelling"}:
            return response
        time.sleep(0.03)
    pytest.fail("The documented graph worker did not reach a terminal state.")


def test_graph_job_discovery_preserves_capability_policy_and_documented_routes(
    client: TestClient,
) -> None:
    """Discover all six protected operations through actual HTTP and the public policy API."""
    response = client.get("/api/studio/capabilities/studio.network_canvas")
    assert response.status_code == 200, response.text
    capability = response.json()
    assert capability["status"] == "experimental"
    assert capability["healthy"] is True
    assert "recoverable background jobs" in capability["summary"]
    assert capability["evidence"] == ["contract_test"]
    assert capability["docs_path"] == "docs/studio/network-canvas.md"
    requirements = {row["name"]: row for row in capability["requirements"]}
    assert requirements["sc_neurocore.studio.graph_job"] == {
        "name": "sc_neurocore.studio.graph_job",
        "available": True,
        "detail": "importable",
    }
    inventory = client.get("/api/studio/capabilities")
    assert inventory.status_code == 200
    assert capability in inventory.json()["capabilities"]

    response = client.get("/openapi.json")
    assert response.status_code == 200, response.text
    schema = response.json()
    policies = build_default_studio_route_policy_registry()
    operation_ids = set()
    for method, path, action in GRAPH_ROUTES:
        operation = schema["paths"][path][method.lower()]
        assert operation["operationId"] not in operation_ids
        operation_ids.add(operation["operationId"])
        assert "200" in operation["responses"]
        policy = policies.policy_for(method, path)
        assert policy.visibility is RouteVisibility.AUTHENTICATED
        assert policy.audit_action == action
        for relative in ["docs/studio/network-canvas.md", "docs/guides/studio.md"]:
            assert f"| {method} | `{path}` |" in (ROOT / relative).read_text()
    parameters = schema["paths"]["/api/graph/jobs"]["post"]["parameters"]
    headers = {row["name"].lower() for row in parameters if row["in"] == "header"}
    assert {"idempotency-key", "x-studio-graph-submission-sha256"} <= headers


def test_documented_graph_job_example_runs_and_retries_through_all_public_routes(
    client: TestClient,
) -> None:
    """Execute the shipped example, retaining exact original result custody across retry."""
    graph = json.loads((ROOT / "examples/studio_graph_job.json").read_text())
    before = client.get("/api/graph/jobs").json()
    assert before["jobs"] == []
    response = client.post("/api/graph/jobs/validate", json=graph)
    assert response.status_code == 200, response.text
    prepared = response.json()
    assert set(prepared) == {
        "schema_version",
        "execution_mode",
        "graph_sha256",
        "request_sha256",
        "submission_sha256",
        "request_seal_sha256",
        "projected_neuron_steps",
        "n_steps",
        "n_neurons",
    }
    assert prepared["schema_version"] == "studio.graph.job.validation.v1"
    assert prepared["execution_mode"] == "async_job"
    assert (prepared["n_neurons"], prepared["n_steps"], prepared["projected_neuron_steps"]) == (
        2,
        200,
        400,
    )
    assert client.get("/api/graph/jobs").json() == before
    headers = {
        "Idempotency-Key": "graph-example-1",
        "X-Studio-Graph-Submission-SHA256": prepared["submission_sha256"],
    }
    response = client.post("/api/graph/jobs", json=graph, headers=headers)
    assert response.status_code == 200, response.text
    admission = response.json()
    assert admission["schema_version"] == "studio.graph.job.v1"
    route = admission["status_route"]
    record = _terminal_response(client, route).json()
    assert record["status"] == "completed", record
    artifact = client.get(route + "/result")
    assert artifact.status_code == 200, artifact.text
    receipt = record["result"]
    assert receipt["schema_version"] == "studio.graph.job.result.v1"
    assert receipt["graph_sha256"] == prepared["graph_sha256"]
    assert receipt["result_sha256"] == hashlib.sha256(artifact.content).hexdigest()
    assert receipt["result_size_bytes"] == len(artifact.content)
    assert artifact.headers["x-studio-artifact-sha256"] == receipt["result_sha256"]
    assert int(artifact.headers["x-studio-artifact-size"]) == len(artifact.content)
    result = artifact.json()
    assert result["schema_version"] == "studio.network-graph-result.v1"
    reference = client.post("/api/graph/simulate", json=graph)
    assert reference.status_code == 200, reference.text
    for field in [
        "populations",
        "topology",
        "n_steps",
        "n_total",
        "n_spikes",
        "spike_times",
        "spike_neurons",
    ]:
        assert result[field] == reference.json()[field], field
    retry = client.post("/api/graph/jobs", json=graph, headers=headers)
    assert retry.status_code == 200
    assert retry.json()["job_id"] == admission["job_id"]
    history = client.get("/api/graph/jobs").json()
    assert history["schema_version"] == "studio.graph.jobs.v1"
    assert [row["job_id"] for row in history["jobs"]] == [admission["job_id"]]
    assert history["jobs"][0]["result"] is None
    cancellation = client.post(route + "/cancel")
    assert cancellation.status_code == 200
    assert cancellation.json()["status"] == "completed"
    assert client.get(route + "/result").content == artifact.content
