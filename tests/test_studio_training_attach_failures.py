# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training weight-attach failure contracts

"""Exercise public warm and live attach failures through real job storage."""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    StudioJobContext,
    StudioJobManager,
    StudioJobRejected,
    StudioRuntimeSettings,
    write_training_weight_checkpoint,
)
from sc_neurocore.studio.platform.training_weights import (
    TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
)

_SOURCE_CONFIG = {"dataset": "synthetic", "hidden": [16]}


def _build_client(tmp_path: Path) -> TestClient:
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit" / "studio.jsonl"),
        job_default_timeout_seconds=120.0,
    )
    return TestClient(create_app(settings), base_url="http://127.0.0.1")


def _manager(client: TestClient) -> StudioJobManager:
    application = cast(Any, client.app)
    return cast(StudioJobManager, application.state.studio_job_manager)


def _weights_bytes() -> bytes:
    torch = pytest.importorskip("torch")
    from sc_neurocore.training import SpikingNet

    model = SpikingNet(n_input=64, n_hidden=16, n_output=10, n_layers=1)
    buffer = io.BytesIO()
    torch.save(
        {
            "config": dict(_SOURCE_CONFIG),
            "final_metrics": {"val_accuracy": 0.5},
            "model_info": {"architecture": "64->16->10"},
            "model_state_dict": model.state_dict(),
            "schema_version": "studio.training.torch-state-dict.v1",
        },
        buffer,
    )
    return buffer.getvalue()


def _parameter_count(weights: bytes) -> int:
    torch = pytest.importorskip("torch")
    checkpoint = torch.load(io.BytesIO(weights), weights_only=True)
    return int(sum(tensor.numel() for tensor in checkpoint["model_state_dict"].values()))


def _detached_summary(tmp_path: Path, weights: bytes) -> dict[str, object]:
    context = StudioJobContext(
        job_id="sj_detached_summary",
        work_dir=tmp_path / "detached-summary",
        cancel_event=threading.Event(),
        max_artifact_bytes=50_000_000,
    )
    return cast(
        dict[str, object],
        write_training_weight_checkpoint(
            context,
            weights_payload=weights,
            config=dict(_SOURCE_CONFIG),
            architecture="64->16->10",
            parameter_count=_parameter_count(weights),
            final_metrics={"val_accuracy": 0.5},
        ).to_public_dict(),
    )


def _submit_source(
    manager: StudioJobManager,
    *,
    weights: bytes,
    publish_artifacts: bool,
    detached_summary: dict[str, object] | None = None,
) -> str:
    def task(context: StudioJobContext) -> dict[str, object]:
        summary = detached_summary
        if publish_artifacts:
            summary = cast(
                dict[str, object],
                write_training_weight_checkpoint(
                    context,
                    weights_payload=weights,
                    config=dict(_SOURCE_CONFIG),
                    architecture="64->16->10",
                    parameter_count=_parameter_count(weights),
                    final_metrics={"val_accuracy": 0.5},
                ).to_public_dict(),
            )
        assert summary is not None
        return {
            "training_status": "completed",
            "final_metrics": {"val_accuracy": 0.5},
            "weight_checkpoint": summary,
        }

    submitted = manager.submit(
        kind="training",
        owner="studio-training",
        request_id=None,
        task=task,
    )
    completed = manager.wait(submitted.job_id, timeout_seconds=10.0)
    assert completed.status == "completed"
    return completed.job_id


def _wait_running(manager: StudioJobManager, job_id: str) -> None:
    deadline = time.monotonic() + 10.0
    while manager.record(job_id).status != "running":
        if time.monotonic() >= deadline:
            pytest.fail("target job did not reach running state")
        time.sleep(0.01)


def _process_target(manager: StudioJobManager, *, seconds: float = 8.0) -> str:
    target = manager.submit_process_task(
        kind="training",
        owner="studio-training",
        request_id=None,
        task_path="tests.studio_job_tasks:process_sleep_task",
        payload={"seconds": seconds},
    )
    _wait_running(manager, target.job_id)
    return target.job_id


def test_warm_attach_distinguishes_missing_and_corrupt_artifacts(tmp_path: Path) -> None:
    """Warm attach maps undeclared and unavailable source artifacts separately."""
    weights = _weights_bytes()
    client = _build_client(tmp_path)
    manager = _manager(client)
    missing_source = _submit_source(
        manager,
        weights=weights,
        publish_artifacts=False,
        detached_summary=_detached_summary(tmp_path, weights),
    )

    missing = client.post(
        "/api/studio/training/weight-restore/attach",
        json={"source_job_id": missing_source, "config": dict(_SOURCE_CONFIG)},
    )

    corrupt_source = _submit_source(manager, weights=weights, publish_artifacts=True)
    metadata_path = tmp_path / "jobs" / corrupt_source / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH
    metadata_path.unlink()
    corrupt = client.post(
        "/api/studio/training/weight-restore/attach",
        json={"source_job_id": corrupt_source, "config": dict(_SOURCE_CONFIG)},
    )

    assert missing.status_code == 404
    assert missing.json()["detail"] == "training_weight_artifact_not_found"
    assert corrupt.status_code == 409
    assert corrupt.json()["detail"] == "training_weight_artifact_unavailable"


def test_live_attach_distinguishes_source_and_artifact_failures(tmp_path: Path) -> None:
    """Live attach returns stable errors for source and artifact failure classes."""
    weights = _weights_bytes()
    client = _build_client(tmp_path)
    manager = _manager(client)
    target_job_id = _process_target(manager)

    missing_source = client.post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": target_job_id, "source_job_id": "sj_missing"},
    )

    undeclared_source = _submit_source(
        manager,
        weights=weights,
        publish_artifacts=False,
        detached_summary=_detached_summary(tmp_path, weights),
    )
    undeclared = client.post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": target_job_id, "source_job_id": undeclared_source},
    )

    corrupt_source = _submit_source(manager, weights=weights, publish_artifacts=True)
    metadata_path = tmp_path / "jobs" / corrupt_source / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH
    metadata_path.unlink()
    corrupt = client.post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": target_job_id, "source_job_id": corrupt_source},
    )
    manager.cancel(target_job_id)
    manager.wait(target_job_id, timeout_seconds=10.0)

    assert missing_source.status_code == 404
    assert missing_source.json()["detail"] == "source_job_not_found"
    assert undeclared.status_code == 404
    assert undeclared.json()["detail"] == "training_weight_artifact_not_found"
    assert corrupt.status_code == 409
    assert corrupt.json()["detail"] == "training_weight_artifact_unavailable"


def test_live_attach_rejects_non_process_target_at_control_delivery(tmp_path: Path) -> None:
    """A running thread target fails closed when process control is unavailable."""
    weights = _weights_bytes()
    client = _build_client(tmp_path)
    manager = _manager(client)
    release = threading.Event()

    def thread_task(_context: StudioJobContext) -> dict[str, object]:
        release.wait(timeout=5.0)
        return {}

    target = manager.submit(
        kind="training",
        owner="studio-training",
        request_id=None,
        task=thread_task,
    )
    _wait_running(manager, target.job_id)
    source_job_id = _submit_source(manager, weights=weights, publish_artifacts=True)

    response = client.post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": target.job_id, "source_job_id": source_job_id},
    )
    release.set()
    manager.wait(target.job_id, timeout_seconds=10.0)

    assert response.status_code == 409
    assert response.json()["detail"] == "training_job_not_running"


def test_live_attach_maps_control_delivery_race_to_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A target stopping during delivery returns the stable conflict contract."""
    weights = _weights_bytes()
    client = _build_client(tmp_path)
    manager = _manager(client)
    target_job_id = _process_target(manager)
    source_job_id = _submit_source(manager, weights=weights, publish_artifacts=True)

    def reject_control(
        _job_id: str,
        *,
        command: Mapping[str, object],
        seed_inputs: Mapping[str, bytes] | None = None,
    ) -> None:
        del command, seed_inputs
        raise StudioJobRejected("Studio job stopped during control delivery.")

    monkeypatch.setattr(manager, "send_control_command", reject_control)
    response = client.post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": target_job_id, "source_job_id": source_job_id},
    )
    manager.cancel(target_job_id)
    manager.wait(target_job_id, timeout_seconds=10.0)

    assert response.status_code == 409
    assert response.json()["detail"] == "training_job_not_running"


def test_live_attach_rejects_registered_architecture_mismatch(tmp_path: Path) -> None:
    """Registered source and target proxies reject different network shapes."""
    pytest.importorskip("torch")
    client = _build_client(tmp_path)
    manager = _manager(client)
    source = client.post(
        "/api/training/start",
        json={
            "dataset": "synthetic",
            "hidden": [16],
            "epochs": 1,
            "batch_size": 1024,
            "timesteps": 1,
        },
    )
    source_job_id = cast(str, source.json()["job_id"])
    source_record = manager.wait(source_job_id, timeout_seconds=120.0)
    assert source_record.status == "completed"

    target = client.post(
        "/api/training/start",
        json={
            "dataset": "synthetic",
            "hidden": [8],
            "epochs": 100_000,
            "batch_size": 1024,
            "timesteps": 1,
        },
    )
    target_job_id = cast(str, target.json()["job_id"])
    _wait_running(manager, target_job_id)

    response = client.post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": target_job_id, "source_job_id": source_job_id},
    )
    manager.cancel(target_job_id)
    manager.wait(target_job_id, timeout_seconds=120.0)

    assert response.status_code == 409
    assert response.json()["detail"] == "architecture_incompatible"
