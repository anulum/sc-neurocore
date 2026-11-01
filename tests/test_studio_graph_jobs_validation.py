# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Nonallocating asynchronous graph identity

"""Resolve async scientific identity through HTTP before reserving any job state."""

import json
import re
from pathlib import Path

from starlette.testclient import TestClient

from sc_neurocore.studio.platform.policy_models import RouteVisibility
from sc_neurocore.studio.platform.policy_routes import build_default_studio_route_policy_registry
from tests.test_studio_graph_jobs import client as client, graph, terminal_record


def test_graph_validation_binds_work_and_distinguishes_layout_from_science(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Repeated real preparation changes neither jobs nor artifacts while identities stay honest."""
    request = graph()
    request["populations"][0]["position"] = {"x": 10.0, "y": 20.0}
    before = set((tmp_path / "jobs").rglob("*"))
    response = client.post("/api/graph/jobs/validate", json=request)
    assert response.status_code == 200, response.text
    identity = response.json()
    assert identity["schema_version"] == "studio.graph.job.validation.v1"
    assert identity["execution_mode"] == "async_job"
    assert (identity["n_steps"], identity["n_neurons"], identity["projected_neuron_steps"]) == (
        200,
        4,
        800,
    )
    for field in ("graph_sha256", "request_sha256", "request_seal_sha256", "submission_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", identity[field])
    request["populations"][0]["position"]["x"] = 11.0
    moved = client.post("/api/graph/jobs/validate", json=request).json()
    assert moved["graph_sha256"] == identity["graph_sha256"]
    assert moved["request_sha256"] != identity["request_sha256"]
    assert moved["request_seal_sha256"] != identity["request_seal_sha256"]
    request["populations"][0]["params"]["tau"] = 11.0
    changed = client.post("/api/graph/jobs/validate", json=request).json()
    assert changed["graph_sha256"] != moved["graph_sha256"]
    assert client.get("/api/graph/jobs").json()["jobs"] == []
    assert set((tmp_path / "jobs").rglob("*")) == before
    assert (
        build_default_studio_route_policy_registry()
        .policy_for("POST", "/api/graph/jobs/validate")
        .visibility
        == RouteVisibility.AUTHENTICATED
    )


def test_graph_validation_uses_async_work_caps_and_no_worker(client: TestClient) -> None:
    """A graph above the synchronous update budget resolves without executing or shortening it."""
    request = graph()
    request["populations"] = [request["populations"][0]]
    request["populations"][0]["count"] = 101
    request.update(projections=[], duration=1000.0)
    response = client.post("/api/graph/jobs/validate", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["projected_neuron_steps"] == 1_010_000
    assert response.json()["n_steps"] == 10_000
    assert client.get("/api/graph/jobs").json()["jobs"] == []
    status = client.get("/api/studio/jobs/status")
    assert status.status_code == 200 and status.json()["process_count"] == 0


def test_graph_validation_refuses_budget_and_nonportable_text_before_allocation(
    client: TestClient,
) -> None:
    """Scientific caps and a real escaped unpaired surrogate are explicit refusals."""
    request = graph()
    request["duration"] = 2_000_001.0
    refused = client.post("/api/graph/jobs/validate", json=request)
    assert refused.status_code == 422 and refused.json()["detail"]["field"] == "duration"
    admission_refusal = client.post("/api/graph/jobs", json=request)
    assert admission_refusal.status_code == 422
    assert admission_refusal.json() == refused.json()
    request = graph()
    request["populations"][0]["label"] = "\ud800"
    refused = client.post(
        "/api/graph/jobs/validate",
        content=json.dumps(request),
        headers={"content-type": "application/json"},
    )
    assert refused.status_code == 422
    assert refused.json()["detail"] == {
        "error": "graph_job_refused",
        "reason": "Graph job request cannot be sealed portably.",
    }
    admission_refusal = client.post(
        "/api/graph/jobs", content=json.dumps(request), headers={"content-type": "application/json"}
    )
    assert admission_refusal.status_code == 422 and admission_refusal.json() == refused.json()
    assert client.get("/api/graph/jobs").json()["jobs"] == []


def test_graph_admission_rechecks_bound_preparation_before_allocating(
    client: TestClient, tmp_path: Path
) -> None:
    """Refuse an obsolete or malformed binding, then admit the newly validated exact input."""
    request = graph()
    response = client.post("/api/graph/jobs/validate", json=request)
    assert response.status_code == 200, response.text
    original = response.json()
    before = set((tmp_path / "jobs").rglob("*"))
    request["seed"] = 43
    refused = client.post(
        "/api/graph/jobs",
        json=request,
        headers={"x-studio-graph-submission-sha256": original["submission_sha256"]},
    )
    assert refused.status_code == 409
    assert refused.json()["detail"] == "graph_job_preparation_changed"
    malformed = client.post(
        "/api/graph/jobs",
        json=request,
        headers={"x-studio-graph-submission-sha256": "not-a-digest"},
    )
    assert malformed.status_code == 422
    assert client.get("/api/graph/jobs").json()["jobs"] == []
    assert set((tmp_path / "jobs").rglob("*")) == before
    current = client.post("/api/graph/jobs/validate", json=request).json()
    accepted = client.post(
        "/api/graph/jobs",
        json=request,
        headers={"x-studio-graph-submission-sha256": current["submission_sha256"]},
    )
    assert accepted.status_code == 200, accepted.text
    receipt = accepted.json()
    assert receipt["graph_sha256"] == current["graph_sha256"] != original["graph_sha256"]
    assert receipt["job"]["experiment_sha256"] == current["submission_sha256"]
    assert terminal_record(client, receipt["status_route"])["status"] == "completed"
