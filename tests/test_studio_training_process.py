# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training process-task contracts

"""Contract tests for the importable Studio training process task."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.training_process import run_training_process_task
from sc_neurocore.studio.platform.training_weights import (
    TRAINING_WEIGHT_ARTIFACT_PATH,
    TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
)

_REAL_TRAINING_CONFIG = {
    "epochs": 1,
    "dataset": "synthetic",
    "batch_size": 1024,
    "hidden": [8],
    "timesteps": 1,
}


def _context(tmp_path: Path, job_id: str) -> StudioJobContext:
    """Return a confined job context for direct process-task tests."""
    work_dir = tmp_path / job_id
    work_dir.mkdir()
    return StudioJobContext(
        job_id=job_id,
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )


def test_training_process_task_writes_terminal_evidence(tmp_path: Path) -> None:
    """Training process task writes route-compatible terminal artifacts."""
    pytest.importorskip("torch")
    context = _context(tmp_path, "sj_training_process")

    result = run_training_process_task(context, dict(_REAL_TRAINING_CONFIG))

    assert result["training_status"] == "completed"
    assert result["final_metrics"] == {
        "train_loss": 0.0,
        "train_accuracy": 0.0,
        "val_loss": 0.0,
        "val_accuracy": 0.0,
    }
    assert [artifact.relative_path for artifact in context.artifacts] == [
        "training/events.jsonl",
        TRAINING_WEIGHT_ARTIFACT_PATH,
        TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
        "training/status.json",
        "training/evidence.json",
    ]
    status_payload = json.loads(
        (tmp_path / "sj_training_process" / "training" / "status.json").read_text()
    )
    evidence_payload = json.loads(
        (tmp_path / "sj_training_process" / "training" / "evidence.json").read_text()
    )
    assert status_payload["job_id"] == "sj_training_process"
    assert status_payload["status"] == "completed"
    assert evidence_payload["schema_version"] == "studio.action-evidence.v1"
    assert evidence_payload["action_kind"] == "studio.training.run"
    assert evidence_payload["evidence_classification"] == "training"
    assert evidence_payload["job_id"] == "sj_training_process"
    assert evidence_payload["replay_route"] == "POST /api/training/start"


def test_training_process_task_publishes_worker_event_log(tmp_path: Path) -> None:
    """Training process task persists child-process events for live tailing."""
    pytest.importorskip("torch")
    context = _context(tmp_path, "sj_training_process_events")

    run_training_process_task(context, dict(_REAL_TRAINING_CONFIG))

    assert [artifact.relative_path for artifact in context.artifacts] == [
        "training/events.jsonl",
        TRAINING_WEIGHT_ARTIFACT_PATH,
        TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
        "training/status.json",
        "training/evidence.json",
    ]
    events = [
        json.loads(line)
        for line in (tmp_path / "sj_training_process_events" / "training" / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [event["event"] for event in events] == ["config", "epoch", "completed"]
    assert events[0]["data"]["dataset"] == "synthetic"
    assert events[0]["data"]["job_id"] == "sj_training_process_events"
    assert events[1]["data"]["train_accuracy"] == 0.0


def test_training_process_task_publishes_weight_checkpoint_metadata(tmp_path: Path) -> None:
    """Training process task publishes terminal weight artifacts when captured."""
    pytest.importorskip("torch")
    context = _context(tmp_path, "sj_training_process_weights")

    result = run_training_process_task(context, dict(_REAL_TRAINING_CONFIG))
    weight_checkpoint = result["weight_checkpoint"]

    assert isinstance(weight_checkpoint, dict)
    assert weight_checkpoint == {
        "architecture": "64->8->10",
        "config_sha256": weight_checkpoint["config_sha256"],
        "final_metrics": {
            "train_accuracy": 0.0,
            "train_loss": 0.0,
            "val_accuracy": 0.0,
            "val_loss": 0.0,
        },
        "format": "torch_state_dict",
        "framework": "pytorch",
        "metadata_artifact": weight_checkpoint["metadata_artifact"],
        "parameter_count": 610,
        "schema_version": "studio.training.weight-checkpoint.v1",
        "weights_artifact": weight_checkpoint["weights_artifact"],
    }
    assert [artifact.relative_path for artifact in context.artifacts] == [
        "training/events.jsonl",
        TRAINING_WEIGHT_ARTIFACT_PATH,
        TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
        "training/status.json",
        "training/evidence.json",
    ]
    metadata = json.loads(
        (
            tmp_path / "sj_training_process_weights" / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH
        ).read_text()
    )
    assert metadata["architecture"] == "64->8->10"
    assert metadata["weights_artifact"]["relative_path"] == TRAINING_WEIGHT_ARTIFACT_PATH


def test_training_process_task_writes_failed_evidence(tmp_path: Path) -> None:
    """Training process task writes failed evidence before propagating errors."""
    pytest.importorskip("torch")
    context = _context(tmp_path, "sj_training_process_failed")

    with pytest.raises(ValueError, match="batch_size"):
        run_training_process_task(
            context,
            {**_REAL_TRAINING_CONFIG, "batch_size": 0},
        )

    evidence_payload = json.loads(
        (tmp_path / "sj_training_process_failed" / "training" / "evidence.json").read_text()
    )
    assert evidence_payload["status"] == "failed"
    assert "batch_size" in evidence_payload["error_message"]
