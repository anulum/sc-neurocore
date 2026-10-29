# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — real hosted job HTTP and worker tests

"""Follow actual HTTP submissions through process workers and sealed artifacts."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import JsonValue
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.bootstrap import bootstrap_studio_admin_identity
from sc_neurocore.studio.platform.jobs_admission import (
    DEFAULT_MAX_CONCURRENT_JOBS,
    DEFAULT_MAX_QUEUED_JOBS,
    StudioJobQueueFull,
)
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from tests.test_studio_host_job_tasks import build_request


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Own a normal application, process manager and bounded isolated test storage."""
    settings = StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs"))
    with TestClient(
        create_app(settings),
        base_url="http://127.0.0.1:8001",
        headers={"Origin": "http://127.0.0.1:5173"},
    ) as connection:
        yield connection


def completed(client: TestClient, job_id: str) -> dict[str, JsonValue]:
    """Read the actual job until completed, failing on any other terminal state."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(f"/api/studio/jobs/{job_id}")
        assert response.status_code == 200
        record: dict[str, JsonValue] = response.json()
        if record["status"] == "completed":
            return record
        assert record["status"] in {"pending", "running"}, record
        time.sleep(0.025)
    pytest.fail("Actual hosted process did not finish")


@pytest.mark.parametrize(
    "operation", ["simulate", "fi_curve", "bifurcation", "heatmap", "sensitivity"]
)
def test_actual_analysis_result_is_sealed_and_downloadable(
    client: TestClient, operation: str
) -> None:
    """All five closed analysis actions reach existing scientific workers over HTTP."""
    request: dict[str, object] = {
        "equations": ["dv/dt = (I-v)/tau"],
        "params": {"tau": 10},
        "init": {"v": 0},
        "duration": 2,
        "dt": 0.1,
    }
    if operation == "fi_curve":
        request.update(i_steps=2, i_min=0, i_max=1)
    elif operation == "bifurcation":
        request.update(sweep_param="tau", sweep_min=5, sweep_max=10, sweep_steps=5)
    elif operation == "heatmap":
        request["params"] = {"tau": 10, "offset": 0}
        request["equations"] = ["dv/dt = (I-v+offset)/tau"]
        request.update(
            param_x="tau",
            x_min=5,
            x_max=10,
            x_steps=3,
            param_y="offset",
            y_min=0,
            y_max=1,
            y_steps=3,
        )
    elif operation == "sensitivity":
        request.update(equations=["dv/dt=I-z*v+a"], params={"z": 1, "a": 2})
    submitted = client.post(
        "/api/studio/host-jobs", json={"operation": operation, "request": request}
    )
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["job_id"]
    record = completed(client, job_id)
    assert record["kind"] == "analysis"
    artifacts = record["artifacts"]
    assert isinstance(artifacts, list)
    result = next(
        a for a in artifacts if isinstance(a, dict) and a["relative_path"] == "host/result.json"
    )
    assert isinstance(result, dict)
    download = client.get(f"/api/studio/jobs/{job_id}/artifacts/host/result.json")
    assert download.status_code == 200
    assert hashlib.sha256(download.content).hexdigest() == result["sha256"]
    assert len(download.content) == result["size_bytes"]
    assert json.loads(download.content) == record["result"]
    if operation == "sensitivity":
        assert [row["param"] for row in download.json()["sensitivities"]] == ["z", "a"]
    # A completed job remains completed when cancellation races with collection.
    cancelled = client.post(f"/api/studio/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "completed"


def test_description_and_fixed_request_refusals(client: TestClient) -> None:
    """Description reaches the real installed adapter and invalid input admits no job."""
    description = client.get("/api/studio/host-bridge")
    assert description.status_code == 200
    assert description.json()["protocol"] == "sc-neurocore.host-api.v1"
    assert description.json()["profile"] == "local-first"
    assert set(description.json()["operations"]) == {
        "simulate",
        "fi_curve",
        "bifurcation",
        "heatmap",
        "sensitivity",
        "ir.build",
        "ir.verify",
        "ir.replay",
        "ir.emit-sv",
    }
    before = client.get("/api/studio/jobs").json()
    for source in [
        '{"operation":"training","request":{}}',
        '{"operation":"simulate","request":{},"task_path":"other"}',
        '{"operation":"simulate","operation":"simulate","request":{"model_name":"LIFNeuron"}}',
    ]:
        response = client.post(
            "/api/studio/host-jobs", content=source, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
    assert client.get("/api/studio/jobs").json() == before
    assert client.post("/api/studio/jobs/sj_0000000000000000/cancel").status_code == 404


def test_general_cancel_keeps_its_administrator_policy(tmp_path: Path) -> None:
    """Normal policy enforcement refuses anonymous and nonadministrator cancellation."""
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"), enforce_route_policies=True
    )
    with TestClient(
        create_app(settings),
        base_url="http://127.0.0.1:8001",
        headers={"Origin": "http://127.0.0.1:5173"},
    ) as client:
        route = "/api/studio/jobs/sj_0000000000000000/cancel"
        assert client.post(route).status_code == 401
        assert client.post(route, headers={"x-studio-principal": "reader"}).status_code == 403


def test_native_compiler_round_trip_uses_real_processes_and_artifacts(client: TestClient) -> None:
    """Build, verify, replay and emit through normal HTTP admission and real workers."""
    graph: str | None = None
    for operation in ["ir.build", "ir.verify", "ir.replay", "ir.emit-sv"]:
        source = build_request() if graph is None else {"ir_text": graph}
        submitted = client.post(
            "/api/studio/host-jobs", json={"operation": operation, "request": source}
        )
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["job_id"]
        record = completed(client, job_id)
        assert record["kind"] == "compiler"
        response = client.get(f"/api/studio/jobs/{job_id}/artifacts/host/result.json")
        assert response.status_code == 200
        outcome = response.json()
        assert outcome["accepted"] is True
        assert outcome["result"]["source_present"] is True
        assert outcome["result"]["source_replay"] == (
            "not_requested" if operation in {"ir.build", "ir.verify"} else "passed"
        )
        graph = outcome["result"]["ir_text"]
        assert isinstance(graph, str)
        artifact = client.get(f"/api/studio/jobs/{job_id}/artifacts/host/graph.scir")
        assert artifact.status_code == 200 and artifact.text == graph
        if operation == "ir.emit-sv":
            rtl = client.get(f"/api/studio/jobs/{job_id}/artifacts/host/output.sv")
            assert rtl.status_code == 200
            assert rtl.text == outcome["result"]["systemverilog"]


def test_running_analysis_can_be_cancelled_and_reaped(client: TestClient) -> None:
    """A real long simulation stops without a completed scientific artifact."""
    response = client.post(
        "/api/studio/host-jobs",
        json={
            "operation": "simulate",
            "request": {
                "equations": ["dv/dt=(I-v)/10"],
                "duration": 100_000,
                "dt": 0.1,
            },
        },
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status = client.get(f"/api/studio/jobs/{job_id}").json()["status"]
        if status == "running":
            break
        assert status == "pending", status
        time.sleep(0.025)
    else:
        pytest.fail("Actual worker never started")
    cancellation = client.post(f"/api/studio/jobs/{job_id}/cancel")
    assert cancellation.status_code == 200
    assert cancellation.json()["status"] in {"cancelling", "cancelled"}
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        record = client.get(f"/api/studio/jobs/{job_id}").json()
        if record["status"] == "cancelled":
            assert record["result"] is None
            assert record["artifacts"] == []
            assert client.post(f"/api/studio/jobs/{job_id}/cancel").json()["status"] == "cancelled"
            return
        assert record["status"] in {"running", "cancelling"}
        time.sleep(0.025)
    pytest.fail("Cancelled worker did not settle")


@pytest.mark.parametrize("enforced", [False, True])
def test_general_cancel_preserves_laboratory_actor_custody(tmp_path: Path, enforced: bool) -> None:
    """Even an administrator cannot use the new route to cancel another actor's fit."""
    from tests.test_studio_fits_routes import _body

    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"), enforce_route_policies=enforced
    )
    with TestClient(
        create_app(settings),
        base_url="http://127.0.0.1:8001",
        headers={"Origin": "http://127.0.0.1:5173"},
    ) as client:
        alice = {"x-studio-principal": "alice", "x-studio-roles": "studio.admin"}
        bob = {"x-studio-principal": "bob", "x-studio-roles": "studio.admin"}
        submitted = client.post(
            "/api/fits/jobs", json=_body(generations=2, population=4), headers=alice
        )
        assert submitted.status_code == 202, submitted.text
        job_id = submitted.json()["job_id"]
        generic = f"/api/studio/jobs/{job_id}/cancel"
        assert client.post(generic, headers=bob).status_code == 404
        if enforced:
            assert client.post(generic).status_code == 401
            allowed = client.post(generic, headers=alice)
        else:
            assert client.post(generic, headers=alice).status_code == 404
            allowed = client.post(f"/api/fits/jobs/{job_id}/cancel")
        assert allowed.status_code == 200
        assert allowed.json()["status"] in {"cancelling", "cancelled", "completed"}


def test_real_production_profile_does_not_offer_local_hosted_execution(tmp_path: Path) -> None:
    """A normally provisioned production app refuses all local-only adapters, even for admin."""
    identity = tmp_path / "identity.json"
    bootstrap = bootstrap_studio_admin_identity(identity)
    settings = StudioRuntimeSettings(
        deployment_profile="production",
        enforce_route_policies=True,
        allow_header_principal=False,
        identity_file_path=str(identity),
        audit_log_path=str(tmp_path / "audit.jsonl"),
        job_root_path=str(tmp_path / "jobs"),
    )
    with TestClient(
        create_app(settings),
        base_url="http://127.0.0.1:8001",
        headers={"Origin": "http://127.0.0.1:5173"},
    ) as client:
        assert client.get("/api/studio/host-bridge").status_code == 404
        auth = {"Authorization": "Bearer " + bootstrap.bearer_token}
        submitted = client.post(
            "/api/studio/host-jobs",
            headers=auth,
            json={
                "operation": "simulate",
                "request": {"equations": ["dv/dt=-v"], "duration": 1},
            },
        )
        assert submitted.status_code == 404
        assert (
            client.post("/api/studio/jobs/sj_0000000000000000/cancel", headers=auth).status_code
            == 404
        )
        assert client.get("/api/health").status_code == 200


def test_actual_shared_admission_pressure_and_unknown_record_refuse_cleanly(tmp_path: Path) -> None:
    """Real peer reservations exhaust capacity; an unknown job cannot claim cancellation."""
    root = tmp_path / "jobs"
    with TestClient(
        create_app(StudioRuntimeSettings(job_root_path=str(root))),
        base_url="http://127.0.0.1:8001",
        headers={"Origin": "http://127.0.0.1:5173"},
    ) as client:
        peer = StudioJobLedger(root=root)
        try:
            admission = SharedJobAdmission(
                peer, max_concurrent=DEFAULT_MAX_CONCURRENT_JOBS, max_queued=DEFAULT_MAX_QUEUED_JOBS
            )

            def reserve(index: int) -> None:
                """Occupy real shared capacity using this live process's supervisor identity."""
                admission.admit(
                    job_id=f"sj_{index:016x}",
                    kind="analysis",
                    actor="capacity-peer",
                    workspace="default",
                    request_id=None,
                    idempotency_key=None,
                    experiment_sha256=None,
                    admission=None,
                    execution_model="process",
                    supervisor=peer.supervisor,
                    timeout_seconds=10,
                )

            for index in range(DEFAULT_MAX_CONCURRENT_JOBS):
                reserve(index)
            with ThreadPoolExecutor(max_workers=DEFAULT_MAX_QUEUED_JOBS) as queue:
                pending = [
                    queue.submit(reserve, index + DEFAULT_MAX_CONCURRENT_JOBS)
                    for index in range(DEFAULT_MAX_QUEUED_JOBS)
                ]
                deadline = time.monotonic() + 5
                while admission.snapshot().queued != DEFAULT_MAX_QUEUED_JOBS:
                    assert time.monotonic() < deadline, "Real peer queue did not fill"
                    time.sleep(0.01)
                refused = client.post(
                    "/api/studio/host-jobs",
                    json={
                        "operation": "simulate",
                        "request": {
                            "equations": ["dv/dt=-v"],
                            "duration": 1,
                        },
                    },
                )
                assert refused.status_code == 409
                assert refused.json() == {"detail": "hosted_job_admission_refused"}
                job_id = "sj_0000000000000000"
                peer.transition(job_id, "unknown")
                cancelled = client.post(f"/api/studio/jobs/{job_id}/cancel")
                assert cancelled.status_code == 409
                assert cancelled.json() == {"detail": "job_cancellation_refused"}
                assert peer.record(job_id).status == "unknown"
                for future in pending:
                    with pytest.raises(StudioJobQueueFull):
                        future.result(timeout=15)
        finally:
            peer.close()
