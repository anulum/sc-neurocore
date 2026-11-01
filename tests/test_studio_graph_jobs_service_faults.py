# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Graph HTTP outcomes after actual storage service faults

"""Damage real service replies after handlers act, retaining actual durable outcomes."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from tests.studio_storage_generation_support import Authority
from tests.test_studio_graph_jobs import graph, terminal_record
from tests.test_studio_graph_jobs_isolated import (
    authority as authority,
    base as base,
    boundary as boundary,
    graph_client as graph_client,
    graph_manager as graph_manager,
    graph_timeout as graph_timeout,
    graph_worker_probe as graph_worker_probe,
    launcher as launcher,
    ledger as ledger,
)


def changed_outcome(payload: bytes, status: str) -> bytes:
    """Rewrite one actual small response to a correlated refusal or unusable record."""
    value = json.loads(payload)
    if status == "pending":
        value["record"]["status"] = "pending"
    else:
        value["status"] = status
        if "record" in value:
            value["record"] = None
        if "artifact" in value:
            value["artifact"] = None
    return json.dumps(value).encode()


@pytest.mark.parametrize("operation", ["admit_named", "record"])
def test_malformed_admission_acknowledgement_recovers_one_actual_job(
    graph_client: TestClient, authority: Authority, ledger: StudioJobLedger, operation: str
) -> None:
    """A damaged admit or post-admit record cannot become 500 or another calculation."""
    authority.alter[operation] = lambda original: b"{}"
    headers = {"idempotency-key": "damaged-admission"}
    response = graph_client.post("/api/graph/jobs", json=graph(), headers=headers)
    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "graph_job_admission_unconfirmed"
    admitted = ledger.list_records()
    assert len(admitted) == 1 and admitted[0].status == "pending"
    retried = graph_client.post("/api/graph/jobs", json=graph(), headers=headers)
    assert retried.status_code == 200, retried.text
    assert retried.json()["job_id"] == admitted[0].job_id
    assert terminal_record(graph_client, retried.json()["status_route"])["status"] == "completed"
    assert len(ledger.list_records()) == 1


@pytest.mark.parametrize(("status", "expected"), [("not_found", 503), ("forbidden", 403)])
def test_post_admission_record_refusal_is_distinct_from_transport_uncertainty(
    graph_client: TestClient,
    authority: Authority,
    ledger: StudioJobLedger,
    status: str,
    expected: int,
) -> None:
    """The route preserves an already admitted job while identifying an explicit refusal."""
    authority.alter["record"] = lambda original: changed_outcome(original, status)
    response = graph_client.post("/api/graph/jobs", json=graph())
    assert response.status_code == expected, response.text
    assert response.json()["detail"] == (
        "graph_job_admission_refused" if expected == 403 else "graph_job_admission_unconfirmed"
    )
    assert len(ledger.list_records()) == 1


def test_actual_storage_capacity_refusal_keeps_standard_queue_detail(
    graph_client: TestClient, authority: Authority, ledger: StudioJobLedger
) -> None:
    """Fill real admission slots with pending records before submitting the graph route."""
    admission = authority.services.admission
    assert admission is not None
    for index in range(2):
        admission.admit(
            job_id=f"sj_{index:016x}",
            kind="analysis",
            actor="studio",
            workspace="default",
            request_id=None,
            idempotency_key=None,
            experiment_sha256=None,
            admission={},
            execution_model="process",
        )
    response = graph_client.post("/api/graph/jobs", json=graph())
    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "job_queue_full"
    assert response.json()["detail"]["running"] == 2
    assert response.json()["detail"]["queued"] == 0
    assert len(ledger.list_records()) == 2


@pytest.mark.parametrize(
    ("status", "expected", "detail"),
    [
        ("forbidden", 404, "graph_job_not_found"),
        ("not_found", 404, "graph_job_not_found"),
        ("pending", 403, "graph_job_cancellation_refused"),
        ("malformed", 503, "graph_job_cancellation_unconfirmed"),
        ("disconnect", 503, "graph_job_cancellation_unconfirmed"),
    ],
)
def test_cancel_maps_real_acted_on_reply_failures(
    graph_client: TestClient, authority: Authority, status: str, expected: int, detail: str
) -> None:
    """An owned real calculation retains actual state despite a damaged cancellation reply."""
    response = graph_client.post("/api/graph/jobs", json=graph())
    assert response.status_code == 200, response.text
    route = response.json()["status_route"]
    # A completed calculation still executes the real cancellation exchange; its
    # terminal result is unchanged even if the final acknowledgement is damaged.
    assert terminal_record(graph_client, route)["status"] == "completed"
    if status == "disconnect":
        authority.lose["cancel"] = 1
    elif status == "malformed":
        authority.alter["cancel"] = lambda original: b"{}"
    else:
        authority.alter["cancel"] = lambda original: changed_outcome(original, status)
    cancelled = graph_client.post(route + "/cancel")
    assert cancelled.status_code == expected, cancelled.text
    assert cancelled.json()["detail"] == detail
    assert graph_client.get(route).json()["status"] == "completed"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("forbidden", 404),
        ("not_found", 404),
        ("unavailable", 409),
        ("malformed", 409),
        ("disconnect", 503),
    ],
)
def test_result_maps_real_service_reply_failures_without_partial_bytes(
    graph_client: TestClient, authority: Authority, status: str, expected: int
) -> None:
    """A real sealed graph is never replaced by an errored or partial scientific download."""
    response = graph_client.post("/api/graph/jobs", json=graph())
    assert response.status_code == 200, response.text
    route = response.json()["status_route"]
    assert terminal_record(graph_client, route)["status"] == "completed"
    original = graph_client.get(route + "/result")
    assert original.status_code == 200 and len(original.content) < 65536
    if status == "disconnect":
        authority.lose["artifact"] = 1
    elif status == "malformed":
        authority.alter["artifact"] = lambda payload: b"{}"
    else:
        authority.alter["artifact"] = lambda payload: changed_outcome(payload, status)
    failed = graph_client.get(route + "/result")
    assert failed.status_code == expected, failed.text
    assert failed.headers.get("content-disposition") is None
    assert graph_client.get(route + "/result").content == original.content


def test_invalid_status_reply_is_an_observation_failure(
    graph_client: TestClient, authority: Authority
) -> None:
    """An actual status transport fault never invents missing state or emits 500."""
    response = graph_client.post("/api/graph/jobs", json=graph())
    assert response.status_code == 200
    route = response.json()["status_route"]
    assert terminal_record(graph_client, route)["status"] == "completed"
    authority.alter["record"] = lambda payload: b"{}"
    failed = graph_client.get(route)
    assert failed.status_code == 503
    assert failed.json()["detail"] == "graph_job_observation_unavailable"
    assert graph_client.get(route).json()["status"] == "completed"
