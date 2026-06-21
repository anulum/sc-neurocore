# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training weight-restore attach tests

"""Tests for the admin training weight-restore warm-start attach endpoint."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    StudioJobContext,
    StudioJobManager,
    StudioRuntimeSettings,
    write_training_weight_checkpoint,
)

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

    return cast(StudioJobManager, client.app.state.studio_job_manager)


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
            weight_checkpoint = write_training_weight_checkpoint(
                context,
                weights_payload=weights_bytes,
                config=dict(_SOURCE_CONFIG),
                architecture="64->16->10",
                parameter_count=int(parameter_count),
                final_metrics={"val_accuracy": 0.5},
            ).to_public_dict()
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
