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
import threading

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import JsonValue, StudioRuntimeSettings
from sc_neurocore.studio.platform.training_checkpoint import (
    STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION,
    build_training_checkpoint,
    import_training_checkpoint_payload,
)
from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.training_weights import write_training_weight_checkpoint


def _weight_checkpoint_metadata(
    tmp_path: Path,
    *,
    config: dict[str, object],
) -> dict[str, JsonValue]:
    """Return writer-produced weight metadata for checkpoint tests."""

    context = StudioJobContext(
        job_id="sj_training",
        work_dir=tmp_path,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )
    return write_training_weight_checkpoint(
        context,
        weights_payload=b"weights",
        config=config,
        architecture="64->128->10",
        parameter_count=9610,
        final_metrics={"train_accuracy": 0.9},
    ).to_public_dict()


def test_training_checkpoint_round_trip_validates_hashes(tmp_path: Path) -> None:
    """Checkpoint import accepts an untampered exported checkpoint."""

    config = {
        "batch_size": 32,
        "dataset": "synthetic",
        "epochs": 4,
        "hidden": [128],
        "surrogate": "superspike",
        "timesteps": 16,
    }
    checkpoint = build_training_checkpoint(
        job_id="sj_training",
        config=config,
        status="completed",
        final_metrics={"train_accuracy": 0.9},
        evidence_summary={
            "action_kind": "studio.training.run",
            "schema_version": "studio.training.evidence-summary.v1",
        },
        weight_checkpoint=_weight_checkpoint_metadata(tmp_path, config=config),
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
    tampered: dict[str, object] = dict(checkpoint)
    tampered["config"] = {"dataset": "synthetic", "epochs": 99}

    with pytest.raises(ValueError, match="config digest mismatch"):
        import_training_checkpoint_payload(tampered)


def test_training_checkpoint_import_rejects_tampered_weight_metadata(
    tmp_path: Path,
) -> None:
    """Checkpoint import rejects invalid weight metadata after export."""

    config = {"dataset": "synthetic", "epochs": 4}
    checkpoint = build_training_checkpoint(
        job_id="sj_training",
        config=config,
        status="completed",
        weight_checkpoint=_weight_checkpoint_metadata(tmp_path, config=config),
        clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
    ).to_public_dict()
    weight_value = checkpoint["weight_checkpoint"]
    assert isinstance(weight_value, dict)
    tampered_weight: dict[str, object] = dict(weight_value)
    tampered_weight["config_sha256"] = "1" * 64
    tampered: dict[str, object] = dict(checkpoint)
    tampered["weight_checkpoint"] = tampered_weight

    with pytest.raises(ValueError, match="config digest mismatch"):
        import_training_checkpoint_payload(tampered)


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
