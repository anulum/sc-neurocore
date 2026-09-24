# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — durable Training Monitor stream tests

"""Exercise the public Training Monitor stream after process-local proxy loss."""

from __future__ import annotations

import json
import hashlib
import threading
import time
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from sc_neurocore.studio.training import get_training_status, stream_metrics


def _event(frame: str) -> dict[str, object]:
    """Decode one SSE frame received from the public training facade."""
    return cast(dict[str, object], json.loads(frame.removeprefix("data: ").strip()))


def test_unregistered_live_training_stream_keeps_tail_and_terminal(tmp_path: Path) -> None:
    """A second API process can tail its durable job until real completion."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=10.0,
    )
    release = threading.Event()

    def task(context: StudioJobContext) -> dict[str, object]:
        context.append_artifact_event(
            "training/events.jsonl",
            {"event": "epoch", "data": {"epoch": 1}, "timestamp": 1.0},
        )
        release.wait(timeout=5.0)
        return {"final_metrics": {"train_accuracy": 0.75}}

    record = manager.submit(kind="training", owner="studio-training", request_id=None, task=task)
    deadline = time.monotonic() + 5.0
    while True:
        payload, _ = manager.read_live_artifact_bytes(
            record.job_id, "training/events.jsonl", offset=0
        )
        if payload:
            break
        if time.monotonic() >= deadline:
            raise AssertionError("training worker did not publish an epoch")
        time.sleep(0.01)

    stream = stream_metrics(record.job_id, manager)
    try:
        assert _event(next(stream)) == {
            "event": "epoch",
            "data": {"epoch": 1},
            "timestamp": 1.0,
        }
        assert _event(next(stream)) == {"event": "heartbeat"}
        release.set()
        assert manager.wait(record.job_id, timeout_seconds=5.0).status == "completed"
        terminal = _event(next(stream))
        assert terminal["event"] == "completed"
        assert terminal["data"] == {"train_accuracy": 0.75}
        assert list(stream) == []
    finally:
        release.set()
        stream.close()


def test_training_status_and_stream_refuse_other_job_kind(tmp_path: Path) -> None:
    """Training routes do not expose a persisted evidence job as training."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training", "evidence"}),
        default_timeout_seconds=10.0,
    )
    record = manager.submit(
        kind="evidence", owner="studio-evidence", request_id=None, task=lambda _context: {}
    )
    assert manager.wait(record.job_id, timeout_seconds=5.0).status == "completed"
    assert get_training_status(record.job_id, manager) == {
        "error": f"Job {record.job_id} not found"
    }
    assert [_event(frame) for frame in stream_metrics(record.job_id, manager)] == [
        {"event": "error", "data": {"message": "Job not found"}}
    ]


def test_restarted_interrupted_training_stream_keeps_recovery_verdict(tmp_path: Path) -> None:
    """An interrupted durable training job never becomes a failed run in SSE."""
    root = tmp_path / "jobs"
    ledger = StudioJobLedger(root=root)
    created = ledger.create(
        job_id="sj_0000000000000001",
        kind="training",
        actor="studio-training",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
    ).record
    ledger.transition(created.job_id, "interrupted", reason="supervisor exited")
    ledger.close()
    manager = StudioJobManager(
        root=root,
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=10.0,
    )

    events = [_event(frame) for frame in stream_metrics(created.job_id, manager)]

    assert len(events) == 1
    assert events[0]["event"] == "interrupted"
    assert events[0]["data"] == {"message": "Training interrupted."}


def test_restarted_http_training_stream_replays_terminal_record(tmp_path: Path) -> None:
    """The public HTTP stream returns a retained training outcome once."""
    settings = StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs"))
    app = create_app(settings)
    manager = cast(StudioJobManager, app.state.studio_job_manager)
    record = manager.submit(
        kind="training",
        owner="studio-training",
        request_id=None,
        task=lambda _context: {"final_metrics": {"val_accuracy": 0.8}},
    )
    assert manager.wait(record.job_id, timeout_seconds=5.0).status == "completed"
    restarted = TestClient(create_app(settings), base_url="http://127.0.0.1")

    response = restarted.get(f"/api/training/stream/{record.job_id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    event = _event(response.text)
    assert event["event"] == "completed"
    assert event["data"] == {"val_accuracy": 0.8}
    assert isinstance(event["timestamp"], float)

    other = manager.submit(
        kind="analysis", owner="studio-analysis", request_id=None, task=lambda _context: {}
    )
    assert manager.wait(other.job_id, timeout_seconds=5.0).status == "completed"
    assert restarted.get(f"/api/training/status/{other.job_id}").status_code == 404
    other_stream = restarted.get(f"/api/training/stream/{other.job_id}")
    assert other_stream.status_code == 200
    assert _event(other_stream.text) == {
        "event": "error",
        "data": {"message": "Job not found"},
    }


def test_training_stream_requires_valid_bearer_under_route_policy(tmp_path: Path) -> None:
    """The browser stream endpoint accepts the same bearer identity as HTTP."""
    identities = tmp_path / "identities.json"
    identities.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "principal_id": "training-viewer",
                        "roles": ["studio.viewer"],
                        "token_sha256": hashlib.sha256(b"training-token").hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        enforce_route_policies=True,
        identity_file_path=str(identities),
        allow_header_principal=False,
    )
    app = create_app(settings)
    manager = cast(StudioJobManager, app.state.studio_job_manager)
    record = manager.submit(
        kind="training", owner="studio-training", request_id=None, task=lambda _context: {}
    )
    assert manager.wait(record.job_id, timeout_seconds=5.0).status == "completed"
    client = TestClient(app, base_url="http://127.0.0.1")
    url = f"/api/training/stream/{record.job_id}"

    assert client.get(url).status_code == 401
    assert client.get(url, headers={"authorization": "Bearer wrong-token"}).status_code == 401
    response = client.get(url, headers={"authorization": "Bearer training-token"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _event(response.text)["event"] == "completed"
