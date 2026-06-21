# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training checkpoint contracts

"""Portable Training Monitor checkpoint manifests for Studio."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypeAlias, cast

from sc_neurocore.studio.platform.evidence_bundle import JsonValue
from sc_neurocore.studio.platform.training_evidence import validate_training_evidence_summary
from sc_neurocore.studio.platform.training_weights import (
    build_training_weight_restore_plan,
    validate_training_weight_checkpoint_metadata,
)

STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION = "studio.training.checkpoint.v1"

TrainingCheckpointConfig: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class StudioTrainingCheckpoint:
    """Path-free manifest for restoring a Studio Training Monitor run.

    Parameters
    ----------
    job_id:
        Source Training Monitor job ID.
    config:
        JSON-serializable training configuration that can seed another Studio
        training run.
    status:
        Source job status at export time.
    final_metrics:
        Terminal metric map, when the source job reached a terminal state.
    evidence_summary:
        Optional path-free terminal evidence summary from the source job.
    weight_checkpoint:
        Optional path-free metadata for a job-managed binary weight artifact.
    generated_at_utc:
        UTC timestamp for the export operation.
    config_sha256:
        SHA-256 digest of the canonical configuration payload.
    checkpoint_sha256:
        SHA-256 digest of the checkpoint payload excluding this digest field.
    """

    job_id: str
    config: TrainingCheckpointConfig
    status: str
    final_metrics: dict[str, JsonValue] | None
    evidence_summary: dict[str, JsonValue] | None
    weight_checkpoint: dict[str, JsonValue] | None
    generated_at_utc: str
    config_sha256: str
    checkpoint_sha256: str
    schema_version: str = STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the JSON-serializable checkpoint payload."""

        payload: dict[str, JsonValue] = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "config": self.config,
            "config_sha256": self.config_sha256,
            "evidence_summary": self.evidence_summary,
            "final_metrics": self.final_metrics,
            "generated_at_utc": self.generated_at_utc,
            "job_id": self.job_id,
            "schema_version": self.schema_version,
            "status": self.status,
            "weight_checkpoint": self.weight_checkpoint,
        }
        return payload


def build_training_checkpoint(
    *,
    job_id: str,
    config: Mapping[str, object],
    status: str,
    final_metrics: Mapping[str, object] | None = None,
    evidence_summary: Mapping[str, object] | None = None,
    weight_checkpoint: Mapping[str, object] | None = None,
    clock: datetime | None = None,
) -> StudioTrainingCheckpoint:
    """Build a portable Training Monitor checkpoint manifest.

    Parameters
    ----------
    job_id:
        Source Training Monitor job ID.
    config:
        Training configuration to preserve.
    status:
        Source job status at export time.
    final_metrics:
        Optional terminal metrics from the training status response.
    evidence_summary:
        Optional path-free terminal evidence summary.
    weight_checkpoint:
        Optional path-free metadata for a job-managed binary weight artifact.
    clock:
        Optional UTC timestamp override for deterministic tests.

    Returns
    -------
    StudioTrainingCheckpoint
        Digest-backed checkpoint manifest suitable for API export.

    Raises
    ------
    ValueError
        If any supplied payload cannot be represented as portable JSON.
    """

    checkpoint_config = _json_object(config, "Training checkpoint config must be JSON.")
    metrics_payload = (
        None
        if final_metrics is None
        else _json_object(final_metrics, "Training checkpoint metrics must be JSON.")
    )
    evidence_payload = (
        None if evidence_summary is None else validate_training_evidence_summary(evidence_summary)
    )
    generated_at = (
        (clock or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    )
    config_sha256 = _sha256_json(checkpoint_config)
    weight_payload = (
        None
        if weight_checkpoint is None
        else validate_training_weight_checkpoint_metadata(
            weight_checkpoint,
            expected_config_sha256=config_sha256,
        )
    )
    base_payload: dict[str, JsonValue] = {
        "config": checkpoint_config,
        "config_sha256": config_sha256,
        "evidence_summary": evidence_payload,
        "final_metrics": metrics_payload,
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
        "job_id": _required_non_empty_string(job_id, "job_id"),
        "schema_version": STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "status": _required_non_empty_string(status, "status"),
        "weight_checkpoint": weight_payload,
    }
    checkpoint_sha256 = _sha256_json(base_payload)
    return StudioTrainingCheckpoint(
        job_id=cast(str, base_payload["job_id"]),
        config=checkpoint_config,
        status=cast(str, base_payload["status"]),
        final_metrics=metrics_payload,
        evidence_summary=evidence_payload,
        weight_checkpoint=weight_payload,
        generated_at_utc=cast(str, base_payload["generated_at_utc"]),
        config_sha256=cast(str, base_payload["config_sha256"]),
        checkpoint_sha256=checkpoint_sha256,
    )


def import_training_checkpoint_payload(
    payload: Mapping[str, object],
) -> dict[str, JsonValue]:
    """Validate a Training Monitor checkpoint import payload.

    Parameters
    ----------
    payload:
        JSON object supplied to ``POST /api/training/checkpoint/import``.

    Returns
    -------
    dict[str, JsonValue]
        Path-free import result containing the restored training config and
        source checkpoint metadata.

    Raises
    ------
    ValueError
        If the checkpoint schema, hashes, or config payload are invalid.
    """

    checkpoint = _json_object(payload, "Training checkpoint import must be JSON.")
    if checkpoint.get("schema_version") != STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Training checkpoint schema is unsupported.")
    config_value = checkpoint.get("config")
    if not isinstance(config_value, dict):
        raise ValueError("Training checkpoint requires a config object.")
    config = _json_object(config_value, "Training checkpoint config must be JSON.")
    expected_config_sha = _required_string_field(checkpoint, "config_sha256")
    if expected_config_sha != _sha256_json(config):
        raise ValueError("Training checkpoint config digest mismatch.")
    weight_value = checkpoint.get("weight_checkpoint")
    weight_checkpoint = (
        None
        if weight_value is None
        else _weight_checkpoint_payload(weight_value, expected_config_sha256=expected_config_sha)
    )
    evidence_value = checkpoint.get("evidence_summary")
    if evidence_value is not None:
        if not isinstance(evidence_value, dict):
            raise ValueError("Training checkpoint evidence must be an object.")
        validate_training_evidence_summary(evidence_value)
    checkpoint_without_digest = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    expected_checkpoint_sha = checkpoint.get("checkpoint_sha256")
    if expected_checkpoint_sha != _sha256_json(checkpoint_without_digest):
        raise ValueError("Training checkpoint digest mismatch.")
    source_job_id = _required_string_field(checkpoint, "job_id")
    source_status = _required_string_field(checkpoint, "status")
    weight_restore_plan = (
        None
        if weight_checkpoint is None
        else build_training_weight_restore_plan(
            source_job_id=source_job_id,
            source_status=source_status,
            weight_checkpoint=weight_checkpoint,
            expected_config_sha256=expected_config_sha,
        ).to_public_dict()
    )
    return {
        "config": config,
        "config_sha256": expected_config_sha,
        "imported_schema_version": STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "source_job_id": source_job_id,
        "source_status": source_status,
        "source_weight_checkpoint": weight_checkpoint,
        "weight_restore_plan": weight_restore_plan,
    }


def _json_object(payload: Mapping[str, object], error_message: str) -> dict[str, JsonValue]:
    """Return a JSON object after recursively validating portable values."""

    return cast(dict[str, JsonValue], _json_value(dict(payload), error_message))


def _json_value(value: object, error_message: str) -> JsonValue:
    """Return a portable JSON value or raise ``ValueError``."""

    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(error_message)
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item, error_message) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(error_message)
            result[key] = _json_value(item, error_message)
        return result
    raise ValueError(error_message)


def _sha256_json(payload: Mapping[str, JsonValue]) -> str:
    """Return the SHA-256 digest of a canonical JSON object."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_non_empty_string(value: str, field_name: str) -> str:
    """Return a required non-empty string value."""

    if not value:
        raise ValueError(f"Training checkpoint requires {field_name}.")
    return value


def _required_string_field(payload: Mapping[str, JsonValue], field_name: str) -> str:
    """Return a required string field from a checkpoint payload."""

    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Training checkpoint requires {field_name}.")
    return value


def _weight_checkpoint_payload(
    value: JsonValue,
    *,
    expected_config_sha256: str,
) -> dict[str, JsonValue]:
    """Validate and return path-free weight checkpoint metadata."""

    if not isinstance(value, dict):
        raise ValueError("Training checkpoint weight metadata must be an object.")
    return validate_training_weight_checkpoint_metadata(
        value,
        expected_config_sha256=expected_config_sha256,
    )


__all__ = [
    "STUDIO_TRAINING_CHECKPOINT_SCHEMA_VERSION",
    "StudioTrainingCheckpoint",
    "TrainingCheckpointConfig",
    "build_training_checkpoint",
    "import_training_checkpoint_payload",
]
