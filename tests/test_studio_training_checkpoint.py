# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training checkpoint tests

"""Tests for portable Studio Training Monitor checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import threading
from typing import cast

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import JsonValue, StudioRuntimeSettings
from sc_neurocore.studio.platform.training_evidence import (
    TRAINING_EVIDENCE_ARTIFACT_PATH,
    TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION,
)
from sc_neurocore.studio.platform.training_checkpoint import (
    STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION,
    build_training_checkpoint,
    import_training_checkpoint_payload,
)
from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.training_weights import (
    STUDIO_TRAINING_WEIGHT_RESTORE_PLAN_SCHEMA_VERSION,
    TRAINING_WEIGHT_ARTIFACT_ROUTE_TEMPLATE,
    write_training_weight_checkpoint,
)


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


def _evidence_summary() -> dict[str, object]:
    """Return a validated Training Monitor evidence summary payload."""

    return {
        "action_kind": "studio.training.run",
        "evidence_artifact": {
            "relative_path": TRAINING_EVIDENCE_ARTIFACT_PATH,
            "sha256": "0" * 64,
            "size_bytes": 128,
        },
        "evidence_classification": "training",
        "job_id": "sj_training",
        "payload_sha256": "1" * 64,
        "replay_route": "POST /api/training/start",
        "result_artifacts": [
            {
                "relative_path": "training/status.json",
                "sha256": "2" * 64,
                "size_bytes": 256,
            }
        ],
        "schema_version": TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
    }


def _rehash_checkpoint(checkpoint: Mapping[str, object]) -> dict[str, object]:
    """Return a checkpoint copy with a matching digest for its current content."""

    updated = dict(checkpoint)
    without_digest = {key: value for key, value in updated.items() if key != "checkpoint_sha256"}
    encoded = json.dumps(without_digest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    updated["checkpoint_sha256"] = hashlib.sha256(encoded).hexdigest()
    return updated


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
        evidence_summary=_evidence_summary(),
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
    assert checkpoint["evidence_summary"] == _evidence_summary()
    restore_plan = imported["weight_restore_plan"]
    assert isinstance(restore_plan, dict)
    assert restore_plan["schema_version"] == STUDIO_TRAINING_WEIGHT_RESTORE_PLAN_SCHEMA_VERSION
    assert restore_plan["artifact_route_template"] == TRAINING_WEIGHT_ARTIFACT_ROUTE_TEMPLATE
    assert restore_plan["source_job_id"] == "sj_training"
    assert restore_plan["source_status"] == "completed"
    weight_checkpoint = checkpoint["weight_checkpoint"]
    assert isinstance(weight_checkpoint, dict)
    assert restore_plan["weights_artifact"] == weight_checkpoint["weights_artifact"]
    assert restore_plan["restore_ready"] is True


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


@pytest.mark.parametrize(
    ("payload", "error_match"),
    [
        ({"schema_version": "studio.old.v1"}, "schema"),
        (
            {
                "schema_version": STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION,
                "config": [],
            },
            "config object",
        ),
        (
            {
                "schema_version": STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION,
                "config": {"dataset": float("nan")},
            },
            "import must be JSON",
        ),
        ({1: "bad"}, "import must be JSON"),
        (
            {"schema_version": STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION, "bad": object()},
            "import must be JSON",
        ),
    ],
)
def test_training_checkpoint_import_rejects_invalid_json_contract(
    payload: dict[object, object],
    error_match: str,
) -> None:
    """Checkpoint import rejects malformed schemas and non-portable JSON."""

    with pytest.raises(ValueError, match=error_match):
        import_training_checkpoint_payload(cast(dict[str, object], payload))


def test_training_checkpoint_rejects_invalid_evidence_summary() -> None:
    """Checkpoint export rejects unverified Training Monitor evidence summaries."""

    with pytest.raises(ValueError, match="evidence summary"):
        build_training_checkpoint(
            job_id="sj_training",
            config={"dataset": "synthetic", "epochs": 4},
            status="completed",
            evidence_summary={"schema_version": TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION},
            clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
        )


def test_training_checkpoint_import_rejects_tampered_evidence_summary() -> None:
    """Checkpoint import validates evidence summaries even when digest matches."""

    checkpoint = build_training_checkpoint(
        job_id="sj_training",
        config={"dataset": "synthetic", "epochs": 4},
        status="completed",
        evidence_summary=_evidence_summary(),
        clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
    ).to_public_dict()
    tampered: dict[str, object] = dict(checkpoint)
    tampered["evidence_summary"] = {
        "schema_version": TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION,
        "status": "unavailable",
    }

    with pytest.raises(ValueError, match="evidence summary"):
        import_training_checkpoint_payload(_rehash_checkpoint(tampered))


def test_training_checkpoint_import_rejects_non_object_evidence_summary() -> None:
    """Checkpoint import requires evidence summary objects."""

    checkpoint = build_training_checkpoint(
        job_id="sj_training",
        config={"dataset": "synthetic", "epochs": 4},
        status="completed",
        clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
    ).to_public_dict()
    tampered: dict[str, object] = dict(checkpoint)
    tampered["evidence_summary"] = []

    with pytest.raises(ValueError, match="evidence"):
        import_training_checkpoint_payload(_rehash_checkpoint(tampered))


def test_training_checkpoint_import_without_weights_has_no_restore_plan() -> None:
    """Checkpoint import omits restore plans when no weight artifact exists."""

    checkpoint = build_training_checkpoint(
        job_id="sj_training",
        config={"dataset": "synthetic", "epochs": 4},
        status="running",
        clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
    ).to_public_dict()

    imported = import_training_checkpoint_payload(checkpoint)

    assert imported["source_weight_checkpoint"] is None
    assert imported["weight_restore_plan"] is None


def test_training_checkpoint_rejects_missing_build_metadata() -> None:
    """Checkpoint export requires source job and status metadata."""

    with pytest.raises(ValueError, match="job_id"):
        build_training_checkpoint(
            job_id="",
            config={"dataset": "synthetic"},
            status="completed",
            clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="status"):
        build_training_checkpoint(
            job_id="sj_training",
            config={"dataset": "synthetic"},
            status="",
            clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
        )


def test_training_checkpoint_import_rejects_missing_source_metadata() -> None:
    """Checkpoint import requires source job and status metadata."""

    checkpoint = build_training_checkpoint(
        job_id="sj_training",
        config={"dataset": "synthetic", "epochs": 4},
        status="completed",
        clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
    ).to_public_dict()
    missing_job_id = dict(checkpoint)
    missing_job_id["job_id"] = ""
    missing_status = dict(checkpoint)
    missing_status["status"] = ""

    with pytest.raises(ValueError, match="job_id"):
        import_training_checkpoint_payload(_rehash_checkpoint(missing_job_id))
    with pytest.raises(ValueError, match="status"):
        import_training_checkpoint_payload(_rehash_checkpoint(missing_status))


def test_training_checkpoint_import_rejects_non_object_weight_metadata() -> None:
    """Checkpoint import requires weight checkpoint objects."""

    checkpoint = build_training_checkpoint(
        job_id="sj_training",
        config={"dataset": "synthetic", "epochs": 4},
        status="completed",
        clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
    ).to_public_dict()
    tampered: dict[str, object] = dict(checkpoint)
    tampered["weight_checkpoint"] = []

    with pytest.raises(ValueError, match="weight metadata"):
        import_training_checkpoint_payload(_rehash_checkpoint(tampered))


def test_training_checkpoint_import_rejects_digest_mismatch() -> None:
    """Checkpoint import rejects stale checkpoint digests."""

    checkpoint = build_training_checkpoint(
        job_id="sj_training",
        config={"dataset": "synthetic", "epochs": 4},
        status="completed",
        clock=datetime(2026, 6, 20, 12, 0, tzinfo=UTC),
    ).to_public_dict()
    tampered: dict[str, object] = dict(checkpoint)
    tampered["status"] = "failed"

    with pytest.raises(ValueError, match="digest mismatch"):
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
    assert import_payload["weight_restore_plan"] is None


def test_training_checkpoint_export_rejects_unknown_job(tmp_path: Path) -> None:
    """Checkpoint export returns not found for unknown Training Monitor jobs."""

    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs")))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/training/checkpoint/missing")

    assert response.status_code == 404
