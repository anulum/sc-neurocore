# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Graph HTTP through the real storage authority and launcher

"""Exercise graph custody through real sockets and workers, without claiming UID isolation."""

from __future__ import annotations

import hashlib
import os
import resource
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.api.candidate_jobs import build_candidate_jobs_router
from sc_neurocore.studio.api.candidates import build_candidates_router
from sc_neurocore.studio.api.graph_jobs import build_graph_jobs_router
from sc_neurocore.studio.api.runtime import build_studio_api_context
from sc_neurocore.studio.api.security import install_studio_security_middleware
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.storage_requester import delegated
from tests.studio_candidate_support import adex_candidate
from tests.studio_storage_generation_runs import (
    api_runtime,
    authority as authority,
    base as base,
    ledger as ledger,
)
from tests.studio_storage_generation_support import FRAME, Authority
from tests.studio_storage_launcher_support import Launcher, SRC, start, shutdown
from tests.studio_graph_worker_probe import (
    GraphWorkerProbe,
    graph_worker_probe as graph_worker_probe,
)
from tests.test_studio_graph_jobs import graph, terminal_record


@pytest.fixture
def launcher(base: Path, graph_worker_probe: GraphWorkerProbe | None) -> Iterator[Launcher]:
    """Run the real launcher, with opt-in observational instrumentation on graph workers."""
    paths = [str(SRC)]
    if graph_worker_probe is not None:
        paths.append(str(graph_worker_probe.directory))
    running = start(base, python_path=paths)
    try:
        yield running
    finally:
        shutdown(running)


@pytest.fixture
def boundary(base: Path) -> StorageBoundaryConfiguration:
    """Use the actual fixture authority's paths and explicit ingress and result ceilings."""
    return StorageBoundaryConfiguration(
        storage_uid=os.getuid(),
        api_uid=os.getuid() + 1,
        worker_uid=os.getuid() + 2,
        authority_root=base / "authority",
        spool_root=base / "spool",
        socket_path=base / "sock/storage.sock",
        workspace="default",
        frame_max_bytes=FRAME,
        max_metadata_bytes=FRAME,
        max_seed_bytes=FRAME,
        max_seed_entries=8,
        max_manifest_bytes=4096,
        max_artifact_bytes=1 << 20,
        max_artifact_entries=64,
        transfer_timeout_seconds=10.0,
        max_connections=4,
    )


@pytest.fixture
def graph_timeout() -> float:
    """Use a trusted operator deadline, overridden explicitly by the real timeout case."""
    return 120.0


@pytest.fixture
def graph_manager(
    base: Path,
    launcher: Launcher,
    authority: Authority,
    boundary: StorageBoundaryConfiguration,
    graph_timeout: float,
) -> IsolatedJobManager:
    """Bind the public facade to the real peer-checked authority and launcher process."""
    assert launcher.process.poll() is None
    return IsolatedJobManager(
        api_runtime(base, authority),
        boundary,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=graph_timeout,
    )


def application(manager: IsolatedJobManager, root: Path) -> FastAPI:
    """Compose production routers and security over the actual isolated service collaborator."""
    app = FastAPI()
    context = build_studio_api_context(
        app,
        StudioRuntimeSettings(
            enforce_route_policies=True,
            job_root_path=str(root / "unused-context-jobs"),
            audit_log_path=str(root / "audit.jsonl"),
            benchmark_databank_path=str(root / "databank.jsonl"),
        ),
    )
    context.studio_job_manager = manager
    app.state.studio_job_manager = manager
    install_studio_security_middleware(app, context)
    app.include_router(build_graph_jobs_router(context))
    app.include_router(build_candidates_router(context))
    app.include_router(build_candidate_jobs_router(context))
    return app


@pytest.fixture
def graph_client(
    graph_manager: IsolatedJobManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Make real authenticated requests; same-UID sockets establish function only."""
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_PROJECT_ROOT", str(tmp_path / "projects"))
    with TestClient(
        application(graph_manager, tmp_path),
        base_url="http://127.0.0.1",
        headers={"x-studio-principal": "alice", "sec-fetch-site": "none"},
    ) as client:
        yield client


def test_isolated_graph_completes_above_sync_budget_and_keeps_actor_custody(
    graph_client: TestClient, graph_manager: IsolatedJobManager
) -> None:
    """A real large graph result crosses storage frames and seals without exposing another actor."""
    request = graph()
    request["populations"] = [request["populations"][0]]
    request["populations"][0]["count"] = 101
    request.update(projections=[], duration=1000.0)
    headers = {"idempotency-key": "isolated-graph-retry"}
    response = graph_client.post("/api/graph/jobs", json=request, headers=headers)
    assert response.status_code == 200, response.text
    receipt = response.json()
    record = terminal_record(graph_client, receipt["status_route"])
    assert record["status"] == "completed", record
    assert record["owner"] == "alice"
    artifact = graph_client.get(receipt["status_route"] + "/result")
    assert artifact.status_code == 200, artifact.text
    result = artifact.json()
    assert record["result"] == {
        "schema_version": "studio.graph.job.result.v1",
        "graph_sha256": receipt["graph_sha256"],
        "result_artifact": "graph/result.json",
        "result_sha256": hashlib.sha256(artifact.content).hexdigest(),
        "result_size_bytes": len(artifact.content),
        "n_steps": result["n_steps"],
        "n_total": result["n_total"],
        "n_spikes": result["n_spikes"],
    }
    assert result["spec"]["graph_sha256"] == receipt["graph_sha256"]
    assert result["spec"]["neuron_steps"] == 1_010_000
    assert result["n_steps"] == 10_000 and result["n_spikes"] > 0
    assert len(artifact.content) > FRAME
    assert artifact.json() == result
    assert (
        hashlib.sha256(artifact.content).hexdigest() == artifact.headers["x-studio-artifact-sha256"]
    )
    assert graph_client.get("/api/graph/jobs").json()["jobs"][0]["result"] is None
    retry = graph_client.post("/api/graph/jobs", json=request, headers=headers)
    assert retry.status_code == 200 and retry.json()["job_id"] == record["job_id"]
    request["seed"] = 43
    conflict = graph_client.post("/api/graph/jobs", json=request, headers=headers)
    assert conflict.status_code == 409, conflict.text
    assert len(graph_client.get("/api/graph/jobs").json()["jobs"]) == 1
    for suffix, method in [("", "GET"), ("/cancel", "POST"), ("/result", "GET")]:
        assert (
            graph_client.request(
                method, receipt["status_route"] + suffix, headers={"x-studio-principal": "bob"}
            ).status_code
            == 404
        )
    assert (
        graph_client.get("/api/graph/jobs", headers={"x-studio-principal": "bob"}).json()["jobs"]
        == []
    )
    with (
        delegated(
            Principal("bob", frozenset()),
            method="GET",
            route="/api/graph/jobs/{job_id}",
            request_id="deny-record",
        ),
        pytest.raises(KeyError),
    ):
        graph_manager.record(record["job_id"])
    with (
        delegated(
            Principal("bob", frozenset()),
            method="GET",
            route="/api/graph/jobs/{job_id}/result",
            request_id="deny-artifact",
        ),
        pytest.raises(KeyError),
    ):
        graph_manager.read_artifact(record["job_id"], "graph/result.json")
    with (
        delegated(
            Principal("alice", frozenset()),
            method="GET",
            route="/api/graph/jobs/{job_id}/result",
            request_id="wrong-artifact-path",
        ),
        pytest.raises(KeyError),
    ):
        graph_manager.read_artifact(record["job_id"], "other.json")
    assert graph_manager.generation_failures == {}


def test_seed_ingress_limit_refuses_before_any_storage_admission(
    graph_client: TestClient, graph_manager: IsolatedJobManager, authority: Authority
) -> None:
    """The public provider's effective seed cap is enforced before even contacting admission."""
    request = graph()
    request["populations"][0]["label"] = "x" * (graph_manager.max_seed_file_bytes + 1)
    before = authority.seen.count("admit_named")
    response = graph_client.post("/api/graph/jobs", json=request)
    assert response.status_code == 422, response.text
    assert authority.seen.count("admit_named") == before
    assert graph_client.get("/api/graph/jobs").json()["jobs"] == []


def test_lost_admission_reply_retries_the_same_durable_graph(
    graph_client: TestClient, authority: Authority, ledger: StudioJobLedger
) -> None:
    """A real relay loses an acted-on reply; the same key recovers exactly one job."""
    authority.lose["admit_named"] = 1
    headers = {"idempotency-key": "lost-graph-reply"}
    lost = graph_client.post("/api/graph/jobs", json=graph(), headers=headers)
    assert lost.status_code == 503
    assert lost.json()["detail"] == "graph_job_admission_unconfirmed"
    admitted_before_retry = ledger.list_records()
    assert len(admitted_before_retry) == 1
    assert admitted_before_retry[0].status == "pending"
    recovered = graph_client.post("/api/graph/jobs", json=graph(), headers=headers)
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["job_id"] == admitted_before_retry[0].job_id
    record = terminal_record(graph_client, recovered.json()["status_route"])
    assert record["status"] == "completed", record
    assert len(graph_client.get("/api/graph/jobs").json()["jobs"]) == 1


def test_graph_routes_refuse_an_actual_candidate_job(
    graph_client: TestClient,
    graph_manager: IsolatedJobManager,
) -> None:
    """Being this actor's analysis job never makes another operation a graph job."""
    candidate = adex_candidate()
    candidate["reference_tests"] = candidate["reference_tests"][:1]
    candidate["reference_tests"][0]["steps"] = 2
    response = graph_client.post(
        "/api/candidates/simulate", json={"candidate": candidate, "steps": 2}
    )
    assert response.status_code == 202, response.text
    record = terminal_record(graph_client, response.json()["status_route"])
    assert record["status"] == "completed", record
    route = "/api/graph/jobs/" + record["job_id"]
    assert graph_client.get(route).status_code == 404
    assert graph_client.post(route + "/cancel").status_code == 404
    assert graph_client.get(route + "/result").status_code == 404
    assert graph_client.get("/api/graph/jobs").json()["jobs"] == []
    with (
        delegated(
            Principal("alice", frozenset()),
            method="POST",
            route="/api/graph/jobs/{job_id}/cancel",
            request_id="wrong-family-cancel",
        ),
        pytest.raises(KeyError),
    ):
        graph_manager.cancel(record["job_id"])


@pytest.mark.parametrize("graph_worker_probe", [True], indirect=True)
def test_running_graph_cancellation_is_authority_owned(
    graph_client: TestClient,
    graph_manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    graph_worker_probe: GraphWorkerProbe | None,
) -> None:
    """Foreign cancellation cannot act, then the owner stops a registered real worker."""
    request = graph()
    request["populations"] = [request["populations"][0]]
    request["populations"][0]["count"] = 2000
    request.update(projections=[], duration=100.0)
    response = graph_client.post("/api/graph/jobs", json=request)
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert graph_worker_probe is not None
    graph_worker_probe.assert_limits(
        {
            resource.RLIMIT_AS: (4 << 30, 4 << 30),
            resource.RLIMIT_CPU: (600, 600),
            resource.RLIMIT_NOFILE: (256, 256),
            resource.RLIMIT_FSIZE: (64 << 20, 64 << 20),
            resource.RLIMIT_CORE: (0, 0),
        }
    )
    assert graph_client.get(receipt["status_route"]).json()["status"] == "running"
    assert (
        graph_client.post(
            receipt["status_route"] + "/cancel", headers={"x-studio-principal": "bob"}
        ).status_code
        == 404
    )
    with (
        delegated(
            Principal("bob", frozenset({"studio.admin"})),
            method="POST",
            route="/api/studio/jobs/{job_id}/cancel",
            request_id="foreign-admin",
        ),
        pytest.raises(KeyError),
    ):
        graph_manager.cancel(receipt["job_id"])
    assert graph_client.post(receipt["status_route"] + "/cancel").status_code == 200
    record = terminal_record(graph_client, receipt["status_route"])
    assert record["status"] == "cancelled", record
    assert record["result"] is None and record["artifacts"] == []
    assert ledger.record(record["job_id"]).status == "cancelled"
    assert graph_manager.generation_failures == {}


def test_isolated_artifact_budget_refuses_before_creating_a_result_file(
    graph_client: TestClient,
    base: Path,
) -> None:
    """Dense output below the global cap cannot overrun the actual one-MiB worker budget."""
    request = graph()
    request["populations"] = [request["populations"][0]]
    request["populations"][0].update(count=100, drive={"kind": "constant", "current": 1e6})
    request.update(projections=[], duration=1000.0, dt=1.0)
    response = graph_client.post("/api/graph/jobs", json=request)
    assert response.status_code == 200, response.text
    receipt = response.json()
    record = terminal_record(graph_client, receipt["status_route"])
    assert record["status"] == "failed", record
    assert record["error"] == "Studio job artifact exceeds configured size limit."
    assert record["result"] is None and record["artifacts"] == []
    assert graph_client.get(receipt["status_route"] + "/result").status_code == 409
    assert list((base / "spool").rglob("graph/result.json")) == []


@pytest.mark.parametrize("graph_timeout", [0.01])
def test_isolated_graph_deadline_stops_and_reaps_its_registered_worker(
    graph_client: TestClient,
    graph_manager: IsolatedJobManager,
    ledger: StudioJobLedger,
) -> None:
    """A real graph worker times out through authority supervision and releases its reservation."""
    response = graph_client.post("/api/graph/jobs", json=graph())
    assert response.status_code == 200, response.text
    receipt = response.json()
    record = terminal_record(graph_client, receipt["status_route"])
    assert record["status"] == "timed_out", record
    assert record["result"] is None and record["artifacts"] == []
    assert graph_client.get(receipt["status_route"] + "/result").status_code == 409
    assert (
        ledger.connection()
        .execute(
            "SELECT COUNT(*) FROM admission_reservations WHERE job_id = ?", (record["job_id"],)
        )
        .fetchone()[0]
        == 0
    )
    assert graph_manager.generation_failures == {}


def test_isolated_graph_validation_respects_seed_budget_without_contacting_admission(
    graph_client: TestClient,
    graph_manager: IsolatedJobManager,
    authority: Authority,
) -> None:
    """Actual isolated preparation resolves identity without making any authority request."""
    response = graph_client.post("/api/graph/jobs/validate", json=graph())
    assert response.status_code == 200, response.text
    assert response.json()["schema_version"] == "studio.graph.job.validation.v1"
    assert authority.seen == []
    oversized = graph()
    oversized["populations"][0]["label"] = "x" * (graph_manager.max_seed_file_bytes + 1)
    refused = graph_client.post("/api/graph/jobs/validate", json=oversized)
    assert refused.status_code == 422
    assert authority.seen == []


def test_a_thousand_distinct_graph_validations_allocate_no_job_state(
    graph_client: TestClient, authority: Authority, ledger: StudioJobLedger, base: Path
) -> None:
    """A sustained real HTTP preparation workload never contacts or grows the job authority."""
    request = graph()
    request["populations"] = [request["populations"][0]]
    request["populations"][0]["count"] = 1
    request.update(projections=[], duration=1.0, dt=1.0)
    before = set((base / "authority").rglob("*"))
    identities: set[str] = set()
    for seed in range(1000):
        request["seed"] = seed
        response = graph_client.post("/api/graph/jobs/validate", json=request)
        assert response.status_code == 200, response.text
        identities.add(response.json()["request_sha256"])
    assert len(identities) == 1000
    assert authority.seen == []
    assert ledger.list_records() == ()
    assert set((base / "authority").rglob("*")) == before
    assert list((base / "spool").iterdir()) == []
