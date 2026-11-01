# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Durable graph HTTP and real worker acceptance

"""Exercise graph admission and complete native process outcomes through HTTP."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

import sc_neurocore
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.network_graph import simulate_graph
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs_ledger_schema import LEDGER_FILENAME
from tests.studio_graph_worker_probe import (
    GraphWorkerProbe,
    graph_worker_probe as graph_worker_probe,
)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Use actual routes and workers with test-owned durable project and job roots."""
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


def graph() -> dict[str, Any]:
    """Supply nondefault parameters, stochastic input, delays and disconnected topology."""
    return {
        "populations": [
            {
                "id": "e",
                "model": "SCLapicqueLIFNeuron",
                "count": 2,
                "neuron_type": "excitatory",
                "params": {"tau": 10.0},
                "drive": {"kind": "constant", "current": 2.0},
            },
            {
                "id": "i",
                "model": "SCLapicqueLIFNeuron",
                "count": 1,
                "neuron_type": "inhibitory",
                "params": {"tau": 15.0},
                "drive": {"kind": "poisson", "rate_hz": 100.0, "weight": 1.0, "seed": 9},
            },
            {
                "id": "disconnected",
                "model": "SCLapicqueLIFNeuron",
                "count": 1,
                "neuron_type": "excitatory",
                "drive": {"kind": "none"},
            },
        ],
        "projections": [
            {
                "id": "ei",
                "source": "e",
                "target": "i",
                "weight": 10.0,
                "delay": 0.2,
                "rule": "random",
                "probability": 0.5,
                "seed": 123,
            },
            {
                "id": "ie",
                "source": "i",
                "target": "e",
                "weight": -8.0,
                "delay": 0.0,
                "rule": "all_to_all",
            },
        ],
        "duration": 20.0,
        "dt": 0.1,
        "seed": 42,
    }


def terminal_record(client: TestClient, route: str) -> dict[str, Any]:
    """Observe a real process to its actual success, failure or cancellation outcome."""
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        response = client.get(route)
        assert response.status_code == 200, response.text
        record = response.json()
        if record["status"] in {
            "completed",
            "failed",
            "cancelled",
            "timed_out",
            "interrupted",
            "unknown",
        }:
            assert record["execution_model"] == "process"
            return dict(record)
        assert record["status"] in {"pending", "running", "cancelling"}, record
        time.sleep(0.03)
    pytest.fail("graph worker did not finish within its test deadline")


def test_http_worker_preserves_complete_public_graph_result(client: TestClient) -> None:
    """Only declared async limits, mode and digest differ from the synchronous result."""
    request = graph()
    synchronous = simulate_graph(request)
    response = client.post("/api/graph/jobs", json=request)
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert receipt["schema_version"] == "studio.graph.job.v1"
    assert receipt["execution_mode"] == "async_job"
    record = terminal_record(client, receipt["status_route"])
    assert record["status"] == "completed", record
    artifact = client.get(receipt["status_route"] + "/result")
    assert artifact.status_code == 200, artifact.text
    actual = artifact.json()
    assert record["result"] == {
        "schema_version": "studio.graph.job.result.v1",
        "graph_sha256": receipt["graph_sha256"],
        "result_artifact": "graph/result.json",
        "result_sha256": hashlib.sha256(artifact.content).hexdigest(),
        "result_size_bytes": len(artifact.content),
        "n_steps": actual["n_steps"],
        "n_total": actual["n_total"],
        "n_spikes": actual["n_spikes"],
    }
    assert actual["spec"]["graph_sha256"] == receipt["graph_sha256"]
    assert actual["spec"]["execution_mode"] == "async_job"
    assert actual["spec"]["limits"] == {
        "max_neurons": 2000,
        "max_steps": 2_000_000,
        "max_neuron_steps": 2_000_000,
    }
    assert synchronous["spec"]["limits"] == {
        "max_neurons": 2000,
        "max_steps": 100_000,
        "max_neuron_steps": 1_000_000,
    }
    assert actual["spec"]["graph_sha256"] != synchronous["spec"]["graph_sha256"]
    assert "execution_mode" not in synchronous["spec"]
    assert set(actual["spec"]) == set(synchronous["spec"]) | {"execution_mode"}
    expected = {
        **synchronous,
        "spec": {
            **synchronous["spec"],
            "limits": actual["spec"]["limits"],
            "execution_mode": "async_job",
            "graph_sha256": actual["spec"]["graph_sha256"],
        },
    }
    assert actual == expected
    assert actual["n_spikes"] > 0
    assert (
        hashlib.sha256(artifact.content).hexdigest() == artifact.headers["x-studio-artifact-sha256"]
    )
    assert int(artifact.headers["x-studio-artifact-size"]) == len(artifact.content)
    assert client.get("/api/graph/jobs").json()["jobs"][0]["result"] is None


@pytest.mark.parametrize(("count", "steps"), [(1, 100_002), (1001, 1001)])
def test_above_synchronous_budgets_execute_without_shortening(
    client: TestClient, count: int, steps: int
) -> None:
    """Actual jobs exceed each sync ceiling while retaining requested clock and events."""
    request = graph()
    request["populations"] = [request["populations"][0]]
    request["populations"][0]["count"] = count
    request.update(projections=[], duration=float(steps), dt=1.0)
    sync = client.post("/api/graph/simulate", json=request)
    assert sync.status_code == 200 and sync.json()["success"] is False
    response = client.post("/api/graph/jobs", json=request)
    assert response.status_code == 200, response.text
    receipt = response.json()
    record = terminal_record(client, receipt["status_route"])
    assert record["status"] == "completed", record
    artifact = client.get(receipt["status_route"] + "/result")
    assert artifact.status_code == 200, artifact.text
    result = artifact.json()
    assert record["result"]["result_sha256"] == hashlib.sha256(artifact.content).hexdigest()
    assert result["n_steps"] == steps
    assert result["duration"] == float(steps)
    assert result["n_total"] == count
    assert result["spec"]["neuron_steps"] == count * steps
    assert result["n_spikes"] > 0
    assert result["n_spikes"] == len(result["spike_times"]) == len(result["spike_neurons"])


def test_job_cap_refusals_allocate_no_record_or_artifact(
    client: TestClient, tmp_path: Path
) -> None:
    """Step, neuron-work, structure and nonfinite quotient refusals keep durable state empty."""
    before = client.get("/api/graph/jobs").json()
    paths_before = set((tmp_path / "jobs").rglob("*"))
    for count, duration in [(1, 2_000_001.0), (2000, 1001.0), (1, 1e308)]:
        request = graph()
        request["populations"] = [request["populations"][0]]
        request["populations"][0]["count"] = count
        request.update(projections=[], duration=duration, dt=0.1 if duration == 1e308 else 1.0)
        refused = client.post("/api/graph/jobs", json=request)
        assert refused.status_code == 422, refused.text
        assert refused.json()["detail"]["field"] == "duration"
    request = graph()
    request["projections"][0]["target"] = "missing"
    assert client.post("/api/graph/jobs", json=request).status_code == 422
    assert client.get("/api/graph/jobs").json() == before
    assert set((tmp_path / "jobs").rglob("*")) == paths_before


def test_exact_retry_returns_same_job_but_changed_input_refuses(client: TestClient) -> None:
    """The public adapter prevents the legacy key-only ledger replay from lying about inputs."""
    request = graph()
    headers = {"idempotency-key": "graph-retry-example"}
    first = client.post("/api/graph/jobs", json=request, headers=headers)
    assert first.status_code == 200, first.text
    original = first.json()
    assert terminal_record(client, original["status_route"])["status"] == "completed"
    second = client.post("/api/graph/jobs", json=request, headers=headers)
    assert second.status_code == 200
    assert second.json()["job_id"] == original["job_id"]
    request["seed"] = 43
    changed = client.post("/api/graph/jobs", json=request, headers=headers)
    assert changed.status_code == 409
    assert len(client.get("/api/graph/jobs").json()["jobs"]) == 1


@pytest.mark.parametrize("delay", [1e308, 10**400])
def test_overflowed_delay_ratio_refuses_before_any_job(
    client: TestClient, delay: float | int
) -> None:
    """Finite JSON values whose delay quotient overflows receive a located refusal."""
    request = graph()
    request["projections"][0]["delay"] = delay
    response = client.post("/api/graph/jobs", json=request)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["field"] == "projections[0].delay"
    assert client.get("/api/graph/jobs").json()["jobs"] == []


def test_host_operation_set_and_analysis_kind_are_unchanged(client: TestClient) -> None:
    """Adding standalone graph work never advertises an unsupported hosted operation."""
    description = client.get("/api/studio/host-bridge")
    assert description.status_code == 200
    assert "graph" not in description.json()["operations"]
    assert (
        client.post(
            "/api/analysis/jobs", json={"analysis": "graph", "payload": graph()}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/studio/host-jobs",
            content=json.dumps({"operation": "graph", "request": graph()}),
            headers={"content-type": "application/json"},
        ).status_code
        == 422
    )


def test_result_size_refusal_preserves_failed_custody_without_partial_export(
    client: TestClient,
) -> None:
    """A genuine dense spike result exceeding sixteen MiB never becomes partial success."""
    request = graph()
    request["populations"] = [request["populations"][0]]
    request["populations"][0].update(count=1000, drive={"kind": "constant", "current": 1e6})
    request.update(projections=[], duration=1000.0, dt=1.0)
    response = client.post("/api/graph/jobs", json=request)
    assert response.status_code == 200, response.text
    receipt = response.json()
    record = terminal_record(client, receipt["status_route"])
    assert record["status"] == "failed", record
    assert record["result"] is None
    assert record["artifacts"] == []
    assert record["error"] == "Graph job JSON exceeds its byte limit."
    assert client.get(receipt["status_route"] + "/result").status_code == 409
    history = client.get("/api/graph/jobs").json()["jobs"]
    assert len(history) == 1 and history[0]["job_id"] == record["job_id"]
    assert history[0]["status"] == "failed"


def test_result_download_refuses_changed_receipts_and_changed_or_missing_bytes(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Real ledger and artifact damage cannot escape as a complete scientific result."""
    response = client.post("/api/graph/jobs", json=graph())
    assert response.status_code == 200, response.text
    submitted = response.json()
    record = terminal_record(client, submitted["status_route"])
    assert record["status"] == "completed", record
    route = submitted["status_route"] + "/result"
    original = client.get(route)
    assert original.status_code == 200
    with sqlite3.connect(tmp_path / "jobs" / LEDGER_FILENAME) as connection:
        row = connection.execute(
            "SELECT result, admission FROM jobs WHERE job_id = ?", (record["job_id"],)
        ).fetchone()
        original_result, original_admission = row
        receipt = json.loads(original_result)
        variants = [
            None,
            {**receipt, "schema_version": "unknown"},
            {**receipt, "graph_sha256": "0" * 64},
            {**receipt, "result_artifact": "../other.json"},
            {**receipt, "result_sha256": "0" * 64},
            {**receipt, "result_size_bytes": True},
            {**receipt, "result_size_bytes": receipt["result_size_bytes"] + 1},
        ]
        for changed in variants:
            connection.execute(
                "UPDATE jobs SET result = ? WHERE job_id = ?",
                (json.dumps(changed), record["job_id"]),
            )
            connection.commit()
            refused = client.get(route)
            assert refused.status_code == 409, refused.text
            assert refused.json()["detail"] == "graph_job_result_unavailable"
        connection.execute(
            "UPDATE jobs SET result = ? WHERE job_id = ?", (original_result, record["job_id"])
        )
        for graph_digest in (None, "invalid-digest"):
            admission = {**json.loads(original_admission), "graph_sha256": graph_digest}
            connection.execute(
                "UPDATE jobs SET admission = ? WHERE job_id = ?",
                (json.dumps(admission), record["job_id"]),
            )
            connection.commit()
            assert client.get(route).status_code == 409
        connection.execute(
            "UPDATE jobs SET admission = ? WHERE job_id = ?", (original_admission, record["job_id"])
        )
        connection.commit()
        manifest = connection.execute(
            "SELECT artifacts FROM jobs WHERE job_id = ?", (record["job_id"],)
        ).fetchone()[0]
        connection.execute("UPDATE jobs SET artifacts = '[]' WHERE job_id = ?", (record["job_id"],))
        connection.commit()
        undeclared = client.get(route)
        assert undeclared.status_code == 404
        assert undeclared.json()["detail"] == "graph_job_result_not_found"
        connection.execute(
            "UPDATE jobs SET artifacts = ? WHERE job_id = ?", (manifest, record["job_id"])
        )
        connection.commit()
    assert client.get(route).content == original.content
    artifacts = list((tmp_path / "jobs").rglob("graph/result.json"))
    assert len(artifacts) == 1
    artifacts[0].write_bytes(original.content + b" ")
    assert client.get(route).status_code == 409
    artifacts[0].write_bytes(original.content)
    assert client.get(route).content == original.content
    artifacts[0].unlink()
    assert client.get(route).status_code in {404, 409}


def test_completed_graph_survives_app_restart_with_exact_result_and_retry_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second real app recovers actor-owned graph custody and the sealed complete result."""
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_PROJECT_ROOT", str(tmp_path / "projects"))
    settings = StudioRuntimeSettings(
        enforce_route_policies=True,
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
        benchmark_databank_path=str(tmp_path / "databank.jsonl"),
    )
    headers = {"x-studio-principal": "alice", "sec-fetch-site": "none"}
    retry_headers = {"idempotency-key": "restart-graph"}
    with TestClient(create_app(settings), base_url="http://127.0.0.1", headers=headers) as first:
        submitted = first.post("/api/graph/jobs", json=graph(), headers=retry_headers)
        assert submitted.status_code == 200, submitted.text
        receipt = submitted.json()
        record = terminal_record(first, receipt["status_route"])
        assert record["status"] == "completed", record
        complete_result = first.get(receipt["status_route"] + "/result")
        assert complete_result.status_code == 200
    with TestClient(create_app(settings), base_url="http://127.0.0.1", headers=headers) as restored:
        recovered = restored.get(receipt["status_route"])
        assert recovered.status_code == 200 and recovered.json() == record
        downloaded = restored.get(receipt["status_route"] + "/result")
        assert downloaded.status_code == 200 and downloaded.content == complete_result.content
        assert hashlib.sha256(downloaded.content).hexdigest() == record["result"]["result_sha256"]
        replayed = restored.post("/api/graph/jobs", json=graph(), headers=retry_headers)
        assert replayed.status_code == 200 and replayed.json()["job_id"] == record["job_id"]
        assert len(restored.get("/api/graph/jobs").json()["jobs"]) == 1
        assert (
            restored.get(receipt["status_route"], headers={"x-studio-principal": "bob"}).status_code
            == 404
        )
        assert (
            restored.get(
                receipt["status_route"] + "/result", headers={"x-studio-principal": "bob"}
            ).status_code
            == 404
        )


def test_embedded_graph_timeout_reaps_worker_and_retains_a_nonexportable_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual embedded process deadline settles graph custody without a readable result."""
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_PROJECT_ROOT", str(tmp_path / "projects"))
    app = create_app(
        StudioRuntimeSettings(
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=0.01,
            audit_log_path=str(tmp_path / "audit.jsonl"),
            benchmark_databank_path=str(tmp_path / "databank.jsonl"),
        )
    )
    with TestClient(
        app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    ) as session:
        submitted = session.post("/api/graph/jobs", json=graph())
        assert submitted.status_code == 200, submitted.text
        receipt = submitted.json()
        record = terminal_record(session, receipt["status_route"])
        assert record["status"] == "timed_out", record
        assert record["result"] is None and record["artifacts"] == []
        assert session.get(receipt["status_route"] + "/result").status_code == 409
        assert app.state.studio_job_manager.status().active_count == 0
        assert app.state.studio_job_manager.unreaped_workers == ()


@pytest.mark.parametrize("graph_worker_probe", [True], indirect=True)
def test_embedded_graph_owner_cancels_a_real_running_worker(
    client: TestClient, graph_worker_probe: GraphWorkerProbe | None
) -> None:
    """The embedded public graph route stops its supervised worker and releases capacity."""
    request = graph()
    request["populations"] = [request["populations"][0]]
    request["populations"][0]["count"] = 2000
    request.update(projections=[], duration=1000.0, dt=1.0)
    submitted = client.post("/api/graph/jobs", json=request)
    assert submitted.status_code == 200, submitted.text
    receipt = submitted.json()
    assert graph_worker_probe is not None
    cpu = math.ceil(StudioRuntimeSettings().job_default_timeout_seconds * (os.cpu_count() or 1))
    data = max(1, os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2)
    graph_worker_probe.assert_limits(
        {
            resource.RLIMIT_DATA: (data, data),
            resource.RLIMIT_CPU: (cpu, cpu + 1),
            resource.RLIMIT_NOFILE: (4096, 4096),
            resource.RLIMIT_FSIZE: (8 * 1024**3, 8 * 1024**3),
            resource.RLIMIT_CORE: (0, 0),
        }
    )
    assert client.get(receipt["status_route"]).json()["status"] == "running"
    cancelled = client.post(receipt["status_route"] + "/cancel")
    assert cancelled.status_code == 200, cancelled.text
    record = terminal_record(client, receipt["status_route"])
    assert record["status"] == "cancelled", record
    assert record["result"] is None and record["artifacts"] == []
    assert client.get(receipt["status_route"] + "/result").status_code == 409


@pytest.mark.parametrize(
    ("fixture", "expected_graph", "expected_result"),
    [
        (
            "test_fixture",
            "926df67d10a4fef7ef382af4e654849ab0bc08de6cbacd388a4b99c314a1286e",
            "c241da2a32bd3b3a8ace8863b84e302ea02e55f645577d01a00e0a8dbfb60715",
        ),
        (
            "example",
            "a2de856760a6c897b963f30c43744284875392eec4670262ae864dd16c5cdbfe",
            "f006bb4848dc1af7b5d08c355fd942f5ce1ff07088ffc0bc7c6898b448ff1c11",
        ),
    ],
)
def test_existing_sync_graph_and_result_digests_are_unchanged(
    fixture: str, expected_graph: str, expected_result: str
) -> None:
    """Pin every scientific result field, checking environment provenance separately.

    Python, NumPy and package versions describe the actual execution environment;
    they cannot be part of an environment-independent scientific golden. The job
    artifact digest still binds the complete result bytes, including provenance.
    """
    request = (
        graph()
        if fixture == "test_fixture"
        else json.loads((Path(__file__).parents[1] / "examples/studio_graph_job.json").read_text())
    )
    result = simulate_graph(request)
    assert result["success"] is True
    assert "execution_mode" not in result["spec"]
    assert result["spec"]["graph_sha256"] == expected_graph
    assert result["execution"].pop("runtime") == {
        "package_version": sc_neurocore.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == expected_result
