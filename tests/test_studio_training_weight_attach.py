# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training weight-restore attach tests

"""Tests for the admin training weight-restore warm-start attach endpoint."""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    STUDIO_CONTROL_COMMAND_FILE,
    STUDIO_CONTROL_DIR,
    STUDIO_CONTROL_SEED_DIR,
    StudioJobContext,
    StudioJobManager,
    StudioRuntimeSettings,
    build_training_weight_restore_plan,
    training_architecture_fingerprint,
    write_training_weight_checkpoint,
)
from sc_neurocore.studio.platform.training_process import run_training_process_task

_SOURCE_CONFIG = {"dataset": "synthetic", "hidden": [16]}


def _build_client(tmp_path: Path) -> TestClient:
    """Return a TestClient backed by a durable Studio job root."""
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit" / "studio.jsonl"),
        job_default_timeout_seconds=120.0,
    )
    app = create_app(settings)
    return TestClient(app, base_url="http://127.0.0.1")


def _job_manager(client: TestClient) -> StudioJobManager:
    """Return the app-local Studio job manager."""
    application = cast(Any, client.app)
    return cast(StudioJobManager, application.state.studio_job_manager)


def _source_weights_bytes(n_output: int) -> bytes:
    """Return a torch checkpoint matching the synthetic-dataset architecture."""
    torch = pytest.importorskip("torch")
    from sc_neurocore.training import SpikingNet

    model = SpikingNet(n_input=64, n_hidden=16, n_output=n_output, n_layers=1)
    payload = {
        "config": dict(_SOURCE_CONFIG),
        "final_metrics": {"val_accuracy": 0.5},
        "model_info": {"architecture": f"64->16->{n_output}"},
        "model_state_dict": model.state_dict(),
        "schema_version": "studio.training.torch-state-dict.v1",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _submit_source_job(
    manager: StudioJobManager,
    *,
    weights_bytes: bytes | None,
) -> str:
    """Submit a completed source training job that publishes weight artifacts."""

    def task(context: StudioJobContext) -> dict[str, object]:
        weight_checkpoint: dict[str, object] | None = None
        if weights_bytes is not None:
            torch = pytest.importorskip("torch")
            parameter_count = sum(
                tensor.numel()
                for tensor in torch.load(io.BytesIO(weights_bytes), weights_only=True)[
                    "model_state_dict"
                ].values()
            )
            weight_checkpoint = cast(
                dict[str, object],
                write_training_weight_checkpoint(
                    context,
                    weights_payload=weights_bytes,
                    config=dict(_SOURCE_CONFIG),
                    architecture="64->16->10",
                    parameter_count=int(parameter_count),
                    final_metrics={"val_accuracy": 0.5},
                ).to_public_dict(),
            )
        return {
            "training_status": "completed",
            "final_metrics": {"val_accuracy": 0.5},
            "weight_checkpoint": weight_checkpoint,
        }

    submitted = manager.submit(
        kind="training",
        owner="studio-training",
        request_id="req-source",
        task=task,
    )
    completed = manager.wait(submitted.job_id, timeout_seconds=10.0)
    assert completed.status == "completed"
    return completed.job_id


def test_attach_warm_start_trains_and_writes_attach_evidence(tmp_path: Path) -> None:
    """A compatible attach warm-starts a job that emits attach evidence."""
    pytest.importorskip("torch")
    client = _build_client(tmp_path)
    manager = _job_manager(client)
    source_job_id = _submit_source_job(manager, weights_bytes=_source_weights_bytes(10))

    response = client.post(
        "/api/studio/training/weight-restore/attach",
        json={
            "source_job_id": source_job_id,
            "config": {"dataset": "synthetic", "hidden": [16], "epochs": 1, "timesteps": 5},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "running"
    assert body["source_job_id"] == source_job_id
    attach_job_id = body["job_id"]

    completed = manager.wait(attach_job_id, timeout_seconds=120.0)
    assert completed.status == "completed", completed.error
    attach_evidence = json.loads(
        manager.read_artifact(attach_job_id, "training/weight-restore-attach.json").payload
    )
    assert attach_evidence["schema_version"] == "studio.training.weight-restore-attach.v1"
    assert attach_evidence["mode"] == "warm_start"
    assert attach_evidence["evidence_classification"] == "training"
    assert attach_evidence["source_job_id"] == source_job_id
    assert attach_evidence["target_job_id"] == attach_job_id
    assert attach_evidence["architecture_fingerprint"] == body["architecture_fingerprint"]


def test_attach_incompatible_architecture_fails_closed(tmp_path: Path) -> None:
    """An architecture mismatch fails the warm-start job before training."""
    pytest.importorskip("torch")
    client = _build_client(tmp_path)
    manager = _job_manager(client)
    # Source weights for a 2-output model cannot load into the synthetic 10-output model.
    source_job_id = _submit_source_job(manager, weights_bytes=_source_weights_bytes(2))

    response = client.post(
        "/api/studio/training/weight-restore/attach",
        json={
            "source_job_id": source_job_id,
            "config": {"dataset": "synthetic", "hidden": [16], "epochs": 1, "timesteps": 5},
        },
    )

    assert response.status_code == 200, response.text
    attach_job_id = response.json()["job_id"]
    completed = manager.wait(attach_job_id, timeout_seconds=120.0)
    # The strict load fails the job before training; the path-free record redacts
    # the detail to the exception type, so no attach evidence is written.
    assert completed.status == "failed"
    assert completed.error == "ValueError"
    with pytest.raises(KeyError):
        manager.read_artifact(attach_job_id, "training/weight-restore-attach.json")


def test_attach_rejects_unknown_job(tmp_path: Path) -> None:
    """The attach endpoint returns 404 for an unknown source training job."""
    client = _build_client(tmp_path)

    response = client.post(
        "/api/studio/training/weight-restore/attach",
        json={"source_job_id": "sj_missing", "config": {}},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "training_job_not_found"


def test_attach_rejects_job_without_weights(tmp_path: Path) -> None:
    """The attach endpoint returns 409 when the source published no weights."""
    client = _build_client(tmp_path)
    manager = _job_manager(client)
    source_job_id = _submit_source_job(manager, weights_bytes=None)

    response = client.post(
        "/api/studio/training/weight-restore/attach",
        json={"source_job_id": source_job_id, "config": {}},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "training_weight_checkpoint_unavailable"


def test_attach_rejects_config_digest_mismatch(tmp_path: Path) -> None:
    """The attach endpoint returns 422 when the expected config digest mismatches."""
    pytest.importorskip("torch")
    client = _build_client(tmp_path)
    manager = _job_manager(client)
    source_job_id = _submit_source_job(manager, weights_bytes=_source_weights_bytes(10))

    response = client.post(
        "/api/studio/training/weight-restore/attach",
        json={
            "source_job_id": source_job_id,
            "config": {"dataset": "synthetic", "hidden": [16]},
            "expected_config_sha256": "0" * 64,
        },
    )

    assert response.status_code == 422


def _control_command_context(
    tmp_path: Path,
    *,
    weights_bytes: bytes,
) -> tuple[StudioJobContext, dict[str, object]]:
    """Return a context seeded with a live attach control command and seeds."""
    work_dir = tmp_path / "live"
    src_dir = tmp_path / "live-src"
    src_ctx = StudioJobContext(
        job_id="sj_src",
        work_dir=src_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=50_000_000,
    )
    summary = write_training_weight_checkpoint(
        src_ctx,
        weights_payload=weights_bytes,
        config=dict(_SOURCE_CONFIG),
        architecture="64->16->10",
        parameter_count=2410,
        final_metrics={"val_accuracy": 0.5},
    ).to_public_dict()
    plan = build_training_weight_restore_plan(
        source_job_id="sj_src",
        source_status="completed",
        weight_checkpoint=summary,
    ).to_public_dict()
    metadata_bytes = (src_dir / "training" / "model_state.json").read_bytes()

    seed_dir = work_dir / STUDIO_CONTROL_SEED_DIR
    seed_dir.mkdir(parents=True)
    (seed_dir / "model_state.pt").write_bytes(weights_bytes)
    (seed_dir / "model_state.json").write_bytes(metadata_bytes)
    context = StudioJobContext(
        job_id="sj_live",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=50_000_000,
    )
    command: dict[str, object] = {
        "action": "attach_weights",
        "restore_plan": plan,
        "architecture_fingerprint": training_architecture_fingerprint(dict(_SOURCE_CONFIG)),
        "weights_seed_path": "model_state.pt",
        "metadata_seed_path": "model_state.json",
    }
    return context, command


def _write_control_command(work_dir: Path, command: dict[str, object]) -> None:
    """Write one operator control command into the production mailbox path."""
    command_dir = work_dir / STUDIO_CONTROL_DIR
    command_dir.mkdir(parents=True, exist_ok=True)
    (command_dir / STUDIO_CONTROL_COMMAND_FILE).write_text(
        json.dumps(command),
        encoding="utf-8",
    )


def test_live_attach_handler_loads_compatible_weights(tmp_path: Path) -> None:
    """A compatible command loads weights during real process-task execution."""
    pytest.importorskip("torch")
    context, command = _control_command_context(tmp_path, weights_bytes=_source_weights_bytes(10))
    _write_control_command(tmp_path / "live", command)

    result = run_training_process_task(
        context,
        {**_SOURCE_CONFIG, "epochs": 1, "batch_size": 1024, "timesteps": 1},
    )

    assert result["training_status"] == "completed"
    evidence = json.loads(
        (tmp_path / "live" / "training" / "weight-restore-attach.json").read_text()
    )
    assert evidence["mode"] == "live"
    assert evidence["target_job_id"] == "sj_live"
    assert (tmp_path / "live" / "training" / "weight-restore-attach.json").is_file()


def test_live_attach_handler_rejects_incompatible_without_crashing(tmp_path: Path) -> None:
    """An incompatible command is rejected while the real run completes."""
    pytest.importorskip("torch")
    context, command = _control_command_context(tmp_path, weights_bytes=_source_weights_bytes(10))
    _write_control_command(tmp_path / "live", command)

    result = run_training_process_task(
        context,
        {
            "dataset": "synthetic",
            "hidden": [8],
            "epochs": 1,
            "batch_size": 1024,
            "timesteps": 1,
        },
    )

    assert result["training_status"] == "completed"
    assert not (tmp_path / "live" / "training" / "weight-restore-attach.json").is_file()


def test_live_attach_poll_ignores_unrelated_command(tmp_path: Path) -> None:
    """The real process task ignores commands that are not weight attaches."""
    pytest.importorskip("torch")
    work_dir = tmp_path / "live"
    (work_dir / STUDIO_CONTROL_DIR).mkdir(parents=True)
    (work_dir / STUDIO_CONTROL_DIR / STUDIO_CONTROL_COMMAND_FILE).write_text(
        json.dumps({"action": "noop"}), encoding="utf-8"
    )
    context = StudioJobContext(
        job_id="sj_live",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=1_048_576,
    )
    result = run_training_process_task(
        context,
        {**_SOURCE_CONFIG, "epochs": 1, "batch_size": 1024, "timesteps": 1},
    )

    assert result["training_status"] == "completed"
    assert not (tmp_path / "live" / "training" / "weight-restore-attach.json").exists()


@pytest.mark.parametrize(
    "command_payload",
    ["{not-json", json.dumps({"action": "attach_weights"})],
)
def test_live_attach_rejects_invalid_control_commands(
    tmp_path: Path,
    command_payload: str,
) -> None:
    """Malformed and incomplete control commands are rejected without job loss."""
    pytest.importorskip("torch")
    work_dir = tmp_path / "live"
    command_dir = work_dir / STUDIO_CONTROL_DIR
    command_dir.mkdir(parents=True)
    (command_dir / STUDIO_CONTROL_COMMAND_FILE).write_text(
        command_payload,
        encoding="utf-8",
    )
    context = StudioJobContext(
        job_id="sj_live",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=50_000_000,
    )

    result = run_training_process_task(
        context,
        {**_SOURCE_CONFIG, "epochs": 1, "batch_size": 1024, "timesteps": 1},
    )

    assert result["training_status"] == "completed"
    assert not (work_dir / "training" / "weight-restore-attach.json").exists()


def _wait_for_running(manager: StudioJobManager, job_id: str) -> None:
    """Block until a process job reaches the running state."""
    deadline = time.monotonic() + 5.0
    while manager.record(job_id).status != "running":
        if time.monotonic() >= deadline:
            pytest.fail("process job did not reach running state")
        time.sleep(0.02)


def test_live_attach_delivers_command_to_running_target(tmp_path: Path) -> None:
    """The live endpoint delivers a control command to a running target job."""
    pytest.importorskip("torch")
    client = _build_client(tmp_path)
    manager = _job_manager(client)
    target = manager.submit_process_task(
        kind="training",
        owner="studio-training",
        request_id="req-target",
        task_path="tests.studio_job_tasks:process_sleep_task",
        payload={"seconds": 4.0},
    )
    _wait_for_running(manager, target.job_id)
    source_job_id = _submit_source_job(manager, weights_bytes=_source_weights_bytes(10))

    response = client.post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": target.job_id, "source_job_id": source_job_id},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "attach_requested"
    assert body["target_job_id"] == target.job_id
    work_dir = tmp_path / "jobs" / target.job_id
    command = json.loads((work_dir / STUDIO_CONTROL_DIR / STUDIO_CONTROL_COMMAND_FILE).read_text())
    assert command["action"] == "attach_weights"
    assert (work_dir / STUDIO_CONTROL_SEED_DIR / "model_state.pt").is_file()


def test_live_attach_rejects_unknown_target(tmp_path: Path) -> None:
    """The live endpoint returns 404 for an unknown target job."""
    client = _build_client(tmp_path)

    response = client.post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": "sj_missing", "source_job_id": "sj_other"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "training_job_not_found"


def test_live_attach_rejects_non_running_target(tmp_path: Path) -> None:
    """The live endpoint returns 409 when the target job is not running."""
    pytest.importorskip("torch")
    client = _build_client(tmp_path)
    manager = _job_manager(client)
    target_job_id = _submit_source_job(manager, weights_bytes=_source_weights_bytes(10))

    response = client.post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": target_job_id, "source_job_id": target_job_id},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "training_job_not_running"


def test_live_attach_rejects_source_without_weights(tmp_path: Path) -> None:
    """The live endpoint returns 409 when the source published no weights."""
    client = _build_client(tmp_path)
    manager = _job_manager(client)
    target = manager.submit_process_task(
        kind="training",
        owner="studio-training",
        request_id="req-target",
        task_path="tests.studio_job_tasks:process_sleep_task",
        payload={"seconds": 3.0},
    )
    _wait_for_running(manager, target.job_id)
    source_job_id = _submit_source_job(manager, weights_bytes=None)

    response = client.post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": target.job_id, "source_job_id": source_job_id},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "training_weight_checkpoint_unavailable"
