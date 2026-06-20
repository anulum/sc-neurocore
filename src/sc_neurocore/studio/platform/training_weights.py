# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training weight checkpoint manifests

"""Path-free metadata contracts for Studio Training Monitor weight artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from sc_neurocore.studio.platform.evidence_bundle import JsonValue
from sc_neurocore.studio.platform.jobs import StudioJobArtifact, StudioJobContext

STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION = "studio.training.weight-checkpoint.v1"
TRAINING_WEIGHT_ARTIFACT_PATH = "training/model_state.pt"
TRAINING_WEIGHT_METADATA_ARTIFACT_PATH = "training/model_state.json"


@dataclass(frozen=True, slots=True)
class StudioTrainingWeightCheckpoint:
    """Path-free metadata for a terminal Training Monitor weight artifact.

    Parameters
    ----------
    framework:
        Framework used to serialize the weight payload.
    format:
        Serialized payload format.
    architecture:
        Human-readable model architecture summary.
    parameter_count:
        Number of serialized model parameters.
    config_sha256:
        SHA-256 digest of the canonical training configuration.
    weights_artifact:
        Manifest entry for the binary weight artifact.
    metadata_artifact:
        Manifest entry for the JSON metadata artifact.
    final_metrics:
        Terminal metric payload associated with the weights.
    schema_version:
        Schema identifier for this metadata contract.
    """

    framework: str
    format: str
    architecture: str
    parameter_count: int
    config_sha256: str
    weights_artifact: StudioJobArtifact
    metadata_artifact: StudioJobArtifact
    final_metrics: dict[str, JsonValue] | None
    schema_version: str = STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return a path-free JSON-compatible checkpoint summary."""

        return {
            "architecture": self.architecture,
            "config_sha256": self.config_sha256,
            "final_metrics": self.final_metrics,
            "format": self.format,
            "framework": self.framework,
            "metadata_artifact": _artifact_public_dict(self.metadata_artifact),
            "parameter_count": self.parameter_count,
            "schema_version": self.schema_version,
            "weights_artifact": _artifact_public_dict(self.weights_artifact),
        }


def write_training_weight_checkpoint(
    context: StudioJobContext,
    *,
    weights_payload: bytes,
    config: Mapping[str, object],
    architecture: str,
    parameter_count: int,
    final_metrics: Mapping[str, object] | None,
) -> StudioTrainingWeightCheckpoint:
    """Write binary training weights and path-free metadata artifacts.

    Parameters
    ----------
    context:
        Confined Studio job context that owns artifact publication.
    weights_payload:
        Serialized weight payload. The context enforces the byte ceiling.
    config:
        Training configuration used to produce the weights.
    architecture:
        Human-readable architecture summary.
    parameter_count:
        Number of model parameters represented by ``weights_payload``.
    final_metrics:
        Terminal training metrics attached to the checkpoint metadata.

    Returns
    -------
    StudioTrainingWeightCheckpoint
        Path-free summary suitable for status and checkpoint API payloads.

    Raises
    ------
    ValueError
        If the payload is empty, metadata is not portable JSON, or the context
        rejects either artifact.
    """

    if not weights_payload:
        raise ValueError("Training weight checkpoint payload is empty.")
    if parameter_count < 0:
        raise ValueError("Training weight checkpoint parameter count is invalid.")
    config_payload = _json_object(config, "Training weight checkpoint config must be JSON.")
    metrics_payload = (
        None
        if final_metrics is None
        else _json_object(final_metrics, "Training weight checkpoint metrics must be JSON.")
    )
    weights_artifact = context.write_artifact(
        TRAINING_WEIGHT_ARTIFACT_PATH,
        weights_payload,
    )
    metadata_payload: dict[str, JsonValue] = {
        "architecture": _required_non_empty_string(architecture, "architecture"),
        "config_sha256": _sha256_json(config_payload),
        "final_metrics": metrics_payload,
        "format": "torch_state_dict",
        "framework": "pytorch",
        "parameter_count": parameter_count,
        "schema_version": STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION,
        "weights_artifact": _artifact_public_dict(weights_artifact),
    }
    metadata_artifact = context.write_artifact(
        TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
        json.dumps(metadata_payload, sort_keys=True),
    )
    return StudioTrainingWeightCheckpoint(
        framework=cast(str, metadata_payload["framework"]),
        format=cast(str, metadata_payload["format"]),
        architecture=cast(str, metadata_payload["architecture"]),
        parameter_count=parameter_count,
        config_sha256=cast(str, metadata_payload["config_sha256"]),
        weights_artifact=weights_artifact,
        metadata_artifact=metadata_artifact,
        final_metrics=metrics_payload,
    )


def _json_object(payload: Mapping[str, object], error_message: str) -> dict[str, JsonValue]:
    """Return a JSON object after recursively validating portable values."""

    return cast(dict[str, JsonValue], _json_value(dict(payload), error_message))


def _artifact_public_dict(artifact: StudioJobArtifact) -> dict[str, JsonValue]:
    """Return a JSON-compatible public artifact manifest entry."""

    return cast(dict[str, JsonValue], artifact.to_public_dict())


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
        raise ValueError(f"Training weight checkpoint requires {field_name}.")
    return value


__all__ = [
    "STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION",
    "TRAINING_WEIGHT_ARTIFACT_PATH",
    "TRAINING_WEIGHT_METADATA_ARTIFACT_PATH",
    "StudioTrainingWeightCheckpoint",
    "write_training_weight_checkpoint",
]
