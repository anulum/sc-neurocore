# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training checkpoint tests

"""Tests for portable Studio Training Monitor checkpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.training_checkpoint import (
    STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION,
    build_training_checkpoint,
    import_training_checkpoint_payload,
)


def test_training_checkpoint_round_trip_validates_hashes() -> None:
    """Checkpoint import accepts an untampered exported checkpoint."""

    checkpoint = build_training_checkpoint(
        job_id="sj_training",
        config={
            "batch_size": 32,
            "dataset": "synthetic",
            "epochs": 4,
            "hidden": [128],
            "surrogate": "superspike",
            "timesteps": 16,
        },
        status="completed",
        final_metrics={"train_accuracy": 0.9},
        evidence_summary={
            "action_kind": "studio.training.run",
            "schema_version": "studio.training.evidence-summary.v1",
        },
        weight_checkpoint={
            "schema_version": "studio.training.weight-checkpoint.v1",
            "weights_artifact": {
                "relative_path": "training/model_state.pt",
                "sha256": "abc",
                "size_bytes": 12,
            },
        },
        clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
    ).to_public_dict()

    imported = import_training_checkpoint_payload(checkpoint)

    assert imported["imported_schema_version"] == STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION
    assert imported["source_job_id"] == "sj_training"
    assert imported["source_status"] == "completed"
    assert imported["source_weight_checkpoint"] == checkpoint["weight_checkpoint"]
    assert imported["config"] == checkpoint["config"]
    assert imported["config_sha256"] == checkpoint["config_sha256"]


def test_training_checkpoint_import_rejects_tampered_config() -> None:
    """Checkpoint import rejects config changes after export."""

    checkpoint = build_training_checkpoint(
        job_id="sj_training",
        config={"dataset": "synthetic", "epochs": 4},
        status="completed",
        clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
    ).to_public_dict()
    tampered = dict(checkpoint)
    tampered["config"] = {"dataset": "synthetic", "epochs": 99}

    with pytest.raises(ValueError, match="config digest mismatch"):
        import_training_checkpoint_payload(cast(dict[str, object], tampered))


def test_training_checkpoint_endpoints_round_trip(tmp_path: Path) -> None:
    """Training checkpoint export and import are wired through FastAPI."""

    settings = StudioRuntimeSettings(
        job_default_timeout_seconds=10.0,
        job_root_path=str(tmp_path / "jobs"),
    )
    app = create_app(settings)
    client = TestClient(app, base_url="http://127.0.0.1")

    started = client.post(
        "/api/training/start",
        json={
            "batch_size": 32,
            "dataset": "synthetic",
            "epochs": 1,
            "surrogate": "atan_surrogate",
            "timesteps": 25,
        },
    )
    assert started.status_code == 200
    job_id = started.json()["job_id"]

    exported = client.get(f"/api/training/checkpoint/{job_id}")
    assert exported.status_code == 200
    checkpoint = exported.json()
    assert checkpoint["schema_version"] == STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["job_id"] == job_id
    assert checkpoint["config"]["dataset"] == "synthetic"
    assert checkpoint["config"]["epochs"] == 1
    assert "checkpoint_sha256" in checkpoint
    assert "weight_checkpoint" in checkpoint

    imported = client.post("/api/training/checkpoint/import", json=checkpoint)
    assert imported.status_code == 200
    import_payload = imported.json()
    assert import_payload["source_job_id"] == job_id
    assert import_payload["config"] == checkpoint["config"]
    assert import_payload["source_weight_checkpoint"] == checkpoint["weight_checkpoint"]


def test_training_checkpoint_export_rejects_unknown_job(tmp_path: Path) -> None:
    """Checkpoint export returns not found for unknown Training Monitor jobs."""

    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs")))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/training/checkpoint/missing")

    assert response.status_code == 404
