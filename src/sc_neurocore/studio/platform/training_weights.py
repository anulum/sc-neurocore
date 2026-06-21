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
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from sc_neurocore.studio.platform.evidence_bundle import JsonValue
from sc_neurocore.studio.platform.jobs import StudioJobArtifact, StudioJobContext

STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION = "studio.training.weight-checkpoint.v1"
STUDIO_TRAINING_WEIGHT_RESTORE_PLAN_SCHEMA_VERSION = "studio.training.weight-restore-plan.v1"
TRAINING_WEIGHT_ARTIFACT_PATH = "training/model_state.pt"
TRAINING_WEIGHT_METADATA_ARTIFACT_PATH = "training/model_state.json"
TRAINING_WEIGHT_ARTIFACT_ROUTE_TEMPLATE = "/api/studio/jobs/{job_id}/artifacts/{artifact_path}"
_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TrainingWeightStateLoader = Callable[[bytes], Mapping[str, object]]


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


@dataclass(frozen=True, slots=True)
class StudioTrainingWeightRestorePlan:
    """Path-free contract for reloading a Training Monitor weight artifact.

    Parameters
    ----------
    source_job_id:
        Studio job ID that owns the published weight artifacts.
    source_status:
        Source training job status reported by the checkpoint import.
    config_sha256:
        SHA-256 digest of the training configuration associated with weights.
    framework:
        Framework expected by the weight payload.
    format:
        Serialized payload format.
    architecture:
        Human-readable model architecture summary.
    parameter_count:
        Number of serialized model parameters.
    weights_artifact:
        Path-free manifest entry for the binary weight artifact.
    metadata_artifact:
        Path-free manifest entry for the JSON metadata artifact.
    artifact_route_template:
        Authenticated artifact download route template for this Studio API.
    loader_policy:
        Required loader trust boundary for clients that materialize weights.
    schema_version:
        Schema identifier for this restore-plan contract.
    """

    source_job_id: str
    source_status: str
    config_sha256: str
    framework: str
    format: str
    architecture: str
    parameter_count: int
    weights_artifact: dict[str, JsonValue]
    metadata_artifact: dict[str, JsonValue]
    artifact_route_template: str = TRAINING_WEIGHT_ARTIFACT_ROUTE_TEMPLATE
    loader_policy: str = "download_from_authenticated_artifact_route_and_verify_sha256"
    schema_version: str = STUDIO_TRAINING_WEIGHT_RESTORE_PLAN_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible restore plan without local paths."""

        return {
            "architecture": self.architecture,
            "artifact_route_template": self.artifact_route_template,
            "config_sha256": self.config_sha256,
            "format": self.format,
            "framework": self.framework,
            "loader_policy": self.loader_policy,
            "metadata_artifact": dict(self.metadata_artifact),
            "parameter_count": self.parameter_count,
            "restore_ready": True,
            "schema_version": self.schema_version,
            "source_job_id": self.source_job_id,
            "source_status": self.source_status,
            "weights_artifact": dict(self.weights_artifact),
        }


@dataclass(frozen=True, slots=True)
class StudioTrainingWeightMaterialization:
    """Trusted in-memory materialization of verified training weights.

    Parameters
    ----------
    source_job_id:
        Studio job ID that owns the source weight artifacts.
    config_sha256:
        SHA-256 digest of the training configuration associated with weights.
    framework:
        Framework used by the trusted loader.
    format:
        Serialized payload format consumed by the trusted loader.
    architecture:
        Human-readable model architecture summary.
    parameter_count:
        Number of parameters declared by the checkpoint metadata.
    state_dict:
        In-memory state dictionary returned by the trusted loader.
    weights_sha256:
        Verified SHA-256 digest of the binary weight payload.
    metadata_sha256:
        Verified SHA-256 digest of the metadata payload.
    schema_version:
        Schema identifier for this materialization contract.
    """

    source_job_id: str
    config_sha256: str
    framework: str
    format: str
    architecture: str
    parameter_count: int
    state_dict: Mapping[str, object]
    weights_sha256: str
    metadata_sha256: str
    schema_version: str = "studio.training.weight-materialization.v1"

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return path-free materialization metadata without tensor payloads."""

        return {
            "architecture": self.architecture,
            "config_sha256": self.config_sha256,
            "format": self.format,
            "framework": self.framework,
            "loaded_key_count": len(self.state_dict),
            "metadata_sha256": self.metadata_sha256,
            "parameter_count": self.parameter_count,
            "schema_version": self.schema_version,
            "source_job_id": self.source_job_id,
            "weights_sha256": self.weights_sha256,
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


def build_training_weight_restore_plan(
    *,
    source_job_id: str,
    source_status: str,
    weight_checkpoint: Mapping[str, object],
    expected_config_sha256: str | None = None,
) -> StudioTrainingWeightRestorePlan:
    """Build a safe restore plan from validated weight metadata.

    Parameters
    ----------
    source_job_id:
        Studio job ID that owns the published training weight artifacts.
    source_status:
        Source training job status associated with the checkpoint.
    weight_checkpoint:
        Path-free weight metadata from a portable training checkpoint.
    expected_config_sha256:
        Optional training config digest that the metadata must match.

    Returns
    -------
    StudioTrainingWeightRestorePlan
        Path-free restore plan that identifies the authenticated artifact route
        and digest checks required before a client materializes weights.

    Raises
    ------
    ValueError
        If source metadata is missing or the weight checkpoint is invalid.
    """

    metadata = validate_training_weight_checkpoint_metadata(
        weight_checkpoint,
        expected_config_sha256=expected_config_sha256,
    )
    weights_artifact = _required_artifact_dict(metadata, "weights_artifact")
    metadata_artifact = _required_artifact_dict(metadata, "metadata_artifact")
    return StudioTrainingWeightRestorePlan(
        source_job_id=_required_non_empty_string(source_job_id, "source_job_id"),
        source_status=_required_non_empty_string(source_status, "source_status"),
        config_sha256=cast(str, metadata["config_sha256"]),
        framework=cast(str, metadata["framework"]),
        format=cast(str, metadata["format"]),
        architecture=cast(str, metadata["architecture"]),
        parameter_count=cast(int, metadata["parameter_count"]),
        weights_artifact=weights_artifact,
        metadata_artifact=metadata_artifact,
    )


def materialize_training_weight_payload(
    *,
    restore_plan: Mapping[str, object],
    metadata_payload: bytes,
    weights_payload: bytes,
    trusted_loader: TrainingWeightStateLoader,
) -> StudioTrainingWeightMaterialization:
    """Validate and materialize a Training Monitor weight payload in memory.

    Parameters
    ----------
    restore_plan:
        ``studio.training.weight-restore-plan.v1`` object produced by a
        validated checkpoint import.
    metadata_payload:
        Raw bytes fetched from the authenticated metadata artifact route.
    weights_payload:
        Raw bytes fetched from the authenticated weight artifact route.
    trusted_loader:
        Loader that deserializes ``weights_payload`` after all schema, size,
        and digest checks pass. Production PyTorch integrations should use a
        loader that restricts deserialization to state dictionaries.

    Returns
    -------
    StudioTrainingWeightMaterialization
        Verified, path-free in-memory materialization metadata plus the loaded
        state dictionary.

    Raises
    ------
    ValueError
        If the restore plan, metadata payload, artifact digests, artifact
        sizes, or loader output is invalid.
    """

    plan = _validate_restore_plan(restore_plan)
    metadata_artifact = _required_artifact_dict(plan, "metadata_artifact")
    weights_artifact = _required_artifact_dict(plan, "weights_artifact")
    _verify_artifact_payload(metadata_payload, metadata_artifact, "metadata_artifact")
    _verify_artifact_payload(weights_payload, weights_artifact, "weights_artifact")

    metadata_json = _metadata_payload_object(metadata_payload)
    metadata_for_validation = dict(metadata_json)
    metadata_for_validation["metadata_artifact"] = dict(metadata_artifact)
    metadata = validate_training_weight_checkpoint_metadata(
        metadata_for_validation,
        expected_config_sha256=cast(str, plan["config_sha256"]),
    )
    if _required_artifact_dict(metadata, "metadata_artifact") != metadata_artifact:
        raise ValueError("Training weight metadata artifact does not match restore plan.")
    if _required_artifact_dict(metadata, "weights_artifact") != weights_artifact:
        raise ValueError("Training weight artifact does not match restore plan.")

    state_dict = _loaded_state_dict(trusted_loader(weights_payload))
    return StudioTrainingWeightMaterialization(
        source_job_id=cast(str, plan["source_job_id"]),
        config_sha256=cast(str, plan["config_sha256"]),
        framework=cast(str, plan["framework"]),
        format=cast(str, plan["format"]),
        architecture=cast(str, plan["architecture"]),
        parameter_count=cast(int, plan["parameter_count"]),
        state_dict=state_dict,
        weights_sha256=cast(str, weights_artifact["sha256"]),
        metadata_sha256=cast(str, metadata_artifact["sha256"]),
    )


def validate_training_weight_checkpoint_metadata(
    payload: Mapping[str, object],
    *,
    expected_config_sha256: str | None = None,
) -> dict[str, JsonValue]:
    """Validate imported Training Monitor weight metadata.

    Parameters
    ----------
    payload:
        Path-free weight metadata from a ``studio.training.checkpoint.v1``
        payload.
    expected_config_sha256:
        Optional checkpoint configuration digest that must match the weight
        metadata configuration digest.

    Returns
    -------
    dict[str, JsonValue]
        JSON-compatible, validated weight metadata.

    Raises
    ------
    ValueError
        If the schema, framework, format, config digest, artifact paths,
        artifact sizes, artifact hashes, or metadata payload are invalid.
    """

    metadata = _json_object(
        payload,
        "Training weight checkpoint metadata must be JSON.",
    )
    if metadata.get("schema_version") != STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Training weight checkpoint schema is unsupported.")
    if metadata.get("framework") != "pytorch":
        raise ValueError("Training weight checkpoint framework is unsupported.")
    if metadata.get("format") != "torch_state_dict":
        raise ValueError("Training weight checkpoint format is unsupported.")
    architecture = metadata.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("Training weight checkpoint requires architecture.")
    parameter_count = metadata.get("parameter_count")
    if not isinstance(parameter_count, int) or parameter_count < 0:
        raise ValueError("Training weight checkpoint parameter count is invalid.")
    config_sha256 = metadata.get("config_sha256")
    if not isinstance(config_sha256, str) or not _SHA256_HEX_PATTERN.fullmatch(config_sha256):
        raise ValueError("Training weight checkpoint config digest is invalid.")
    if expected_config_sha256 is not None and config_sha256 != expected_config_sha256:
        raise ValueError("Training weight checkpoint config digest mismatch.")
    _validate_artifact_metadata(
        metadata.get("weights_artifact"),
        expected_path=TRAINING_WEIGHT_ARTIFACT_PATH,
        field_name="weights_artifact",
    )
    _validate_artifact_metadata(
        metadata.get("metadata_artifact"),
        expected_path=TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
        field_name="metadata_artifact",
    )
    final_metrics = metadata.get("final_metrics")
    if final_metrics is not None and not isinstance(final_metrics, dict):
        raise ValueError("Training weight checkpoint metrics must be an object.")
    return metadata


def _validate_restore_plan(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    """Return a validated weight restore plan."""

    plan = _json_object(payload, "Training weight restore plan must be JSON.")
    if plan.get("schema_version") != STUDIO_TRAINING_WEIGHT_RESTORE_PLAN_SCHEMA_VERSION:
        raise ValueError("Training weight restore plan schema is unsupported.")
    if plan.get("loader_policy") != "download_from_authenticated_artifact_route_and_verify_sha256":
        raise ValueError("Training weight restore plan loader policy is unsupported.")
    if plan.get("artifact_route_template") != TRAINING_WEIGHT_ARTIFACT_ROUTE_TEMPLATE:
        raise ValueError("Training weight restore plan route template is unsupported.")
    for field_name in (
        "source_job_id",
        "source_status",
        "config_sha256",
        "framework",
        "format",
        "architecture",
    ):
        _required_json_string(plan, field_name)
    if not _SHA256_HEX_PATTERN.fullmatch(_required_json_string(plan, "config_sha256")):
        raise ValueError("Training weight restore plan config digest is invalid.")
    parameter_count = plan.get("parameter_count")
    if not isinstance(parameter_count, int) or parameter_count < 0:
        raise ValueError("Training weight restore plan parameter count is invalid.")
    if plan.get("framework") != "pytorch":
        raise ValueError("Training weight restore plan framework is unsupported.")
    if plan.get("format") != "torch_state_dict":
        raise ValueError("Training weight restore plan format is unsupported.")
    _validate_artifact_metadata(
        plan.get("weights_artifact"),
        expected_path=TRAINING_WEIGHT_ARTIFACT_PATH,
        field_name="weights_artifact",
    )
    _validate_artifact_metadata(
        plan.get("metadata_artifact"),
        expected_path=TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
        field_name="metadata_artifact",
    )
    return plan


def _verify_artifact_payload(
    payload: bytes,
    artifact: Mapping[str, JsonValue],
    field_name: str,
) -> None:
    """Verify one artifact payload against its path-free manifest entry."""

    size_bytes = artifact.get("size_bytes")
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError(f"Training weight {field_name} size is invalid.")
    if len(payload) != size_bytes:
        raise ValueError(f"Training weight {field_name} size mismatch.")
    sha256 = artifact.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX_PATTERN.fullmatch(sha256):
        raise ValueError(f"Training weight {field_name} digest is invalid.")
    if _sha256_bytes(payload) != sha256:
        raise ValueError(f"Training weight {field_name} digest mismatch.")


def _metadata_payload_object(payload: bytes) -> dict[str, JsonValue]:
    """Decode a portable weight metadata payload."""

    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Training weight metadata payload is invalid JSON.") from exc
    if not isinstance(decoded, dict):
        raise ValueError("Training weight metadata payload must be a JSON object.")
    return _json_object(
        cast(Mapping[str, object], decoded),
        "Training weight metadata payload must be JSON.",
    )


def _loaded_state_dict(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Validate trusted loader output as a state dictionary."""

    state_dict = dict(payload)
    for key in state_dict:
        if not isinstance(key, str) or not key:
            raise ValueError("Training weight loader returned an invalid state key.")
    return state_dict


def _json_object(payload: Mapping[str, object], error_message: str) -> dict[str, JsonValue]:
    """Return a JSON object after recursively validating portable values."""

    return cast(dict[str, JsonValue], _json_value(dict(payload), error_message))


def _artifact_public_dict(artifact: StudioJobArtifact) -> dict[str, JsonValue]:
    """Return a JSON-compatible public artifact manifest entry."""

    return cast(dict[str, JsonValue], artifact.to_public_dict())


def _required_artifact_dict(
    metadata: Mapping[str, JsonValue],
    field_name: str,
) -> dict[str, JsonValue]:
    """Return one validated artifact metadata object from a checkpoint."""

    value = metadata.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"Training weight checkpoint requires {field_name}.")
    return cast(dict[str, JsonValue], dict(value))


def _validate_artifact_metadata(
    value: object,
    *,
    expected_path: str,
    field_name: str,
) -> None:
    """Validate one path-free artifact manifest entry."""

    if not isinstance(value, dict):
        raise ValueError(f"Training weight checkpoint requires {field_name}.")
    artifact = _json_object(value, f"Training weight checkpoint {field_name} must be JSON.")
    if artifact.get("relative_path") != expected_path:
        raise ValueError(f"Training weight checkpoint {field_name} path is invalid.")
    size_bytes = artifact.get("size_bytes")
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError(f"Training weight checkpoint {field_name} size is invalid.")
    sha256 = artifact.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX_PATTERN.fullmatch(sha256):
        raise ValueError(f"Training weight checkpoint {field_name} digest is invalid.")


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


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of a byte payload."""

    return hashlib.sha256(payload).hexdigest()


def _required_json_string(payload: Mapping[str, JsonValue], field_name: str) -> str:
    """Return a required non-empty string from a JSON object."""

    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Training weight restore plan requires {field_name}.")
    return value


def _required_non_empty_string(value: str, field_name: str) -> str:
    """Return a required non-empty string value."""

    if not value:
        raise ValueError(f"Training weight checkpoint requires {field_name}.")
    return value


__all__ = [
    "STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION",
    "STUDIO_TRAINING_WEIGHT_RESTORE_PLAN_SCHEMA_VERSION",
    "TRAINING_WEIGHT_ARTIFACT_ROUTE_TEMPLATE",
    "TRAINING_WEIGHT_ARTIFACT_PATH",
    "TRAINING_WEIGHT_METADATA_ARTIFACT_PATH",
    "StudioTrainingWeightCheckpoint",
    "StudioTrainingWeightMaterialization",
    "StudioTrainingWeightRestorePlan",
    "TrainingWeightStateLoader",
    "build_training_weight_restore_plan",
    "materialize_training_weight_payload",
    "validate_training_weight_checkpoint_metadata",
    "write_training_weight_checkpoint",
]
