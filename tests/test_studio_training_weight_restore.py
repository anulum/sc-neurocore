# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training weight-restore endpoint tests

"""Tests for the admin training weight-restore materialization endpoint."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
    StudioJobContext,
    StudioJobManager,
    StudioRuntimeSettings,
    write_training_weight_checkpoint,
)

_CONFIG = {"dataset": "synthetic", "epochs": 2}


def _build_client(tmp_path: Path) -> TestClient:
    """Return a TestClient backed by a durable Studio job root."""

    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit" / "studio.jsonl"),
        job_default_timeout_seconds=10.0,
    )
    app = create_app(settings)
    return TestClient(app, base_url="http://127.0.0.1")


def _job_manager(client: TestClient) -> StudioJobManager:
    """Return the app-local Studio job manager."""

    return cast(StudioJobManager, client.app.state.studio_job_manager)


def _torch_weights_bytes(config: dict[str, object] | None = None) -> bytes:
    """Return a portable torch checkpoint payload like the Training Monitor."""

    torch = pytest.importorskip("torch")
    payload = {
        "config": dict(config or _CONFIG),
        "final_metrics": {"train_accuracy": 0.75},
        "model_info": {"architecture": "64->10"},
        "model_state_dict": {
            "fc.weight": torch.zeros(2, 3),
            "fc.bias": torch.zeros(2),
        },
        "schema_version": STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
    }
    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _submit_training_source(
    manager: StudioJobManager,
    *,
    weights_bytes: bytes,
    config: dict[str, object],
    publish_weights: bool = True,
) -> str:
    """Submit a completed training job that publishes weight artifacts."""

    def task(context: StudioJobContext) -> dict[str, object]:
        weight_checkpoint: dict[str, object] | None = None
        if publish_weights:
            weight_checkpoint = write_training_weight_checkpoint(
                context,
                weights_payload=weights_bytes,
                config=config,
                architecture="64->10",
                parameter_count=8,
                final_metrics={"train_accuracy": 0.75},
            ).to_public_dict()
        return {
            "training_status": "completed",
            "final_metrics": {"train_accuracy": 0.75},
            "weight_checkpoint": weight_checkpoint,
        }

    submitted = manager.submit(
        kind="training",
        owner="studio-training",
        request_id="req-train",
        task=task,
    )
    completed = manager.wait(submitted.job_id, timeout_seconds=5.0)
    assert completed.status == "completed"
    return completed.job_id


def test_weight_restore_materializes_completed_training_job(tmp_path: Path) -> None:
    """Endpoint materializes verified weights and emits training evidence."""

    pytest.importorskip("torch")
    client = _build_client(tmp_path)
    manager = _job_manager(client)
    weights_bytes = _torch_weights_bytes()
    source_job_id = _submit_training_source(
        manager,
        weights_bytes=weights_bytes,
        config=_CONFIG,
    )

    response = client.post(
        "/api/studio/training/weight-restore",
        json={"source_job_id": source_job_id},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == "studio.training.weight-restore.v1"
    assert body["evidence_classification"] == "training"
    assert body["status"] == "completed"
    assert body["source_job_id"] == source_job_id
    assert body["source_status"] == "completed"
    materialization = body["materialization"]
    assert materialization["schema_version"] == "studio.training.weight-materialization.v1"
    assert materialization["loaded_key_count"] == 2
    assert materialization["weights_sha256"] == hashlib.sha256(weights_bytes).hexdigest()
    artifact_paths = [artifact["relative_path"] for artifact in body["artifacts"]]
    assert "training/weight-restore.json" in artifact_paths


def test_weight_restore_rejects_unknown_job(tmp_path: Path) -> None:
    """Endpoint returns 404 for an unknown source training job."""

    client = _build_client(tmp_path)

    response = client.post(
        "/api/studio/training/weight-restore",
        json={"source_job_id": "sj_missing"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "training_job_not_found"


def test_weight_restore_rejects_job_without_weights(tmp_path: Path) -> None:
    """Endpoint returns 409 when the training job published no weights."""

    client = _build_client(tmp_path)
    manager = _job_manager(client)
    source_job_id = _submit_training_source(
        manager,
        weights_bytes=b"",
        config=_CONFIG,
        publish_weights=False,
    )

    response = client.post(
        "/api/studio/training/weight-restore",
        json={"source_job_id": source_job_id},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "training_weight_checkpoint_unavailable"


def test_weight_restore_rejects_config_digest_mismatch(tmp_path: Path) -> None:
    """Endpoint returns 422 when the expected config digest does not match."""

    pytest.importorskip("torch")
    client = _build_client(tmp_path)
    manager = _job_manager(client)
    source_job_id = _submit_training_source(
        manager,
        weights_bytes=_torch_weights_bytes(),
        config=_CONFIG,
    )

    response = client.post(
        "/api/studio/training/weight-restore",
        json={"source_job_id": source_job_id, "expected_config_sha256": "0" * 64},
    )

    assert response.status_code == 422


def test_weight_restore_evidence_feeds_evidence_bundle(tmp_path: Path) -> None:
    """Restore evidence is accepted by the admin evidence bundle endpoint."""

    pytest.importorskip("torch")
    client = _build_client(tmp_path)
    manager = _job_manager(client)
    source_job_id = _submit_training_source(
        manager,
        weights_bytes=_torch_weights_bytes(),
        config=_CONFIG,
    )
    restore = client.post(
        "/api/studio/training/weight-restore",
        json={"source_job_id": source_job_id},
    ).json()
    restore_evidence = {
        key: value for key, value in restore.items() if key not in {"job_id", "artifacts"}
    }

    bundle = client.post(
        "/api/studio/evidence/bundle",
        json={"weight_restore_results": [restore_evidence], "include_audit": False},
    )

    assert bundle.status_code == 200, bundle.text
    entries = bundle.json()["manifest"]["entries"]
    restore_entries = [
        entry for entry in entries if entry["type"] == "training_weight_restore_result"
    ]
    assert len(restore_entries) == 1
    assert restore_entries[0]["evidence_classification"] == "training"
