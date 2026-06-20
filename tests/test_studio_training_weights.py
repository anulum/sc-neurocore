# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training weight checkpoint tests

"""Tests for Studio Training Monitor weight checkpoint artifacts."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.training_weights import (
    STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION,
    TRAINING_WEIGHT_ARTIFACT_PATH,
    TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
    write_training_weight_checkpoint,
)


def _context(tmp_path: Path) -> StudioJobContext:
    """Return a confined Studio job context for weight checkpoint tests."""

    return StudioJobContext(
        job_id="sj_weights",
        work_dir=tmp_path,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )


def test_write_training_weight_checkpoint_publishes_binary_and_metadata(
    tmp_path: Path,
) -> None:
    """Weight checkpoint writer emits path-free manifest metadata."""

    context = _context(tmp_path)
    weights_payload = b"serialized weights"

    summary = write_training_weight_checkpoint(
        context,
        weights_payload=weights_payload,
        config={"dataset": "synthetic", "epochs": 2},
        architecture="64->128->10",
        parameter_count=9610,
        final_metrics={"train_accuracy": 0.75},
    ).to_public_dict()

    assert summary["schema_version"] == STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION
    assert summary["framework"] == "pytorch"
    assert summary["format"] == "torch_state_dict"
    assert summary["parameter_count"] == 9610
    assert summary["weights_artifact"] == {
        "relative_path": TRAINING_WEIGHT_ARTIFACT_PATH,
        "sha256": hashlib.sha256(weights_payload).hexdigest(),
        "size_bytes": len(weights_payload),
    }
    assert [artifact.relative_path for artifact in context.artifacts] == [
        TRAINING_WEIGHT_ARTIFACT_PATH,
        TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
    ]
    metadata = json.loads((tmp_path / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_text())
    assert metadata["schema_version"] == STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION
    assert metadata["weights_artifact"] == summary["weights_artifact"]


def test_write_training_weight_checkpoint_rejects_invalid_payload(
    tmp_path: Path,
) -> None:
    """Weight checkpoint writer rejects empty or non-portable metadata inputs."""

    context = _context(tmp_path)

    with pytest.raises(ValueError, match="empty"):
        write_training_weight_checkpoint(
            context,
            weights_payload=b"",
            config={"dataset": "synthetic"},
            architecture="64->10",
            parameter_count=1,
            final_metrics=None,
        )
    with pytest.raises(ValueError, match="metrics"):
        write_training_weight_checkpoint(
            context,
            weights_payload=b"weights",
            config={"dataset": "synthetic"},
            architecture="64->10",
            parameter_count=1,
            final_metrics={"bad": float("nan")},
        )
