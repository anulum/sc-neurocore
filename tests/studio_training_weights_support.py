# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training weights test support

"""Shared fixtures for Studio training weight checkpoint and restore tests."""

from __future__ import annotations

import hashlib

import json

import threading

from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext

from sc_neurocore.studio.platform.training_weight_loader import (
    load_training_weight_state_dict,
)

from sc_neurocore.studio.platform.training_weights import (
    STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
    STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION,
    STUDIO_TRAINING_WEIGHT_RESTORE_ATTACH_SCHEMA_VERSION,
    STUDIO_TRAINING_WEIGHT_RESTORE_PLAN_SCHEMA_VERSION,
    STUDIO_TRAINING_WEIGHT_RESTORE_SCHEMA_VERSION,
    TRAINING_WEIGHT_ARTIFACT_ROUTE_TEMPLATE,
    TRAINING_WEIGHT_ARTIFACT_PATH,
    TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
    build_training_weight_restore_attach_evidence,
    build_training_weight_restore_evidence,
    build_training_weight_restore_plan,
    materialize_training_weight_payload,
    training_architecture_fingerprint,
    validate_training_weight_checkpoint_metadata,
    validate_training_weight_restore_attach_evidence,
    validate_training_weight_restore_evidence,
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

def _torch_checkpoint_bytes(
    *,
    schema_version: str = STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
    state_dict: dict[str, object] | None = None,
    include_state_dict: bool = True,
) -> bytes:
    """Return a portable torch checkpoint payload like the Training Monitor."""

    from io import BytesIO

    torch = pytest.importorskip("torch")
    payload: dict[str, object] = {
        "config": {"dataset": "synthetic", "epochs": 2},
        "final_metrics": {"train_accuracy": 0.75},
        "model_info": {"architecture": "64->10"},
        "schema_version": schema_version,
    }
    if include_state_dict:
        if state_dict is None:
            state_dict = {
                "fc.weight": torch.zeros(2, 3),
                "fc.bias": torch.zeros(2),
            }
        payload["model_state_dict"] = state_dict
    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()

def _materialization(tmp_path: Path):
    """Return a verified materialization for restore-evidence tests."""

    context = _context(tmp_path)
    summary = write_training_weight_checkpoint(
        context,
        weights_payload=b"weights",
        config={"dataset": "synthetic", "epochs": 2},
        architecture="64->10",
        parameter_count=650,
        final_metrics={"train_accuracy": 0.75},
    ).to_public_dict()
    plan = build_training_weight_restore_plan(
        source_job_id="sj_training",
        source_status="completed",
        weight_checkpoint=summary,
        expected_config_sha256=str(summary["config_sha256"]),
    ).to_public_dict()
    return materialize_training_weight_payload(
        restore_plan=plan,
        metadata_payload=(tmp_path / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_bytes(),
        weights_payload=(tmp_path / TRAINING_WEIGHT_ARTIFACT_PATH).read_bytes(),
        trusted_loader=lambda payload: {"layer.weight": payload},
    )

__all__ = [
    "annotations",
    "hashlib",
    "json",
    "threading",
    "Path",
    "pytest",
    "StudioJobContext",
    "load_training_weight_state_dict",
    "STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION",
    "STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION",
    "STUDIO_TRAINING_WEIGHT_RESTORE_ATTACH_SCHEMA_VERSION",
    "STUDIO_TRAINING_WEIGHT_RESTORE_PLAN_SCHEMA_VERSION",
    "STUDIO_TRAINING_WEIGHT_RESTORE_SCHEMA_VERSION",
    "TRAINING_WEIGHT_ARTIFACT_ROUTE_TEMPLATE",
    "TRAINING_WEIGHT_ARTIFACT_PATH",
    "TRAINING_WEIGHT_METADATA_ARTIFACT_PATH",
    "build_training_weight_restore_attach_evidence",
    "build_training_weight_restore_evidence",
    "build_training_weight_restore_plan",
    "materialize_training_weight_payload",
    "training_architecture_fingerprint",
    "validate_training_weight_checkpoint_metadata",
    "validate_training_weight_restore_attach_evidence",
    "validate_training_weight_restore_evidence",
    "write_training_weight_checkpoint",
    "_context",
    "_torch_checkpoint_bytes",
    "_materialization",
]
