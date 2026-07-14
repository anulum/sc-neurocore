# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training weight-attach orchestration

"""Orchestrate warm-start and live attachment of verified training weights."""

from __future__ import annotations

from typing import Any, cast

from sc_neurocore.studio._training_control import (
    _get_registered_pair,
    _get_training_status,
    _register_job,
)
from sc_neurocore.studio._training_job import TrainingJob
from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifactUnavailable,
    StudioJobManager,
    StudioJobRejected,
)
from sc_neurocore.studio.platform.training_weights import (
    STUDIO_TRAINING_WEIGHT_RESTORE_ATTACH_OWNER,
    TRAINING_WEIGHT_ARTIFACT_PATH,
    TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
    build_training_weight_restore_plan,
    training_architecture_fingerprint,
)

_LIVE_ATTACH_WEIGHTS_SEED = "model_state.pt"
_LIVE_ATTACH_METADATA_SEED = "model_state.json"


def _start_training_attach(
    source_job_id: str,
    config: dict[str, Any],
    job_manager: StudioJobManager,
    *,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Start a warm-start job seeded with verified source weights."""
    from sc_neurocore.studio.platform.training_process import (
        TRAINING_ATTACH_PROCESS_TASK,
        TRAINING_ATTACH_SEED_METADATA_PATH,
        TRAINING_ATTACH_SEED_WEIGHTS_PATH,
    )

    status_payload = _get_training_status(source_job_id, job_manager)
    if "status" not in status_payload:
        return {"error": "training_job_not_found"}
    weight_checkpoint = status_payload.get("weight_checkpoint")
    if not isinstance(weight_checkpoint, dict):
        return {"error": "training_weight_checkpoint_unavailable"}
    source_status = cast(str, status_payload["status"])

    restore_plan = build_training_weight_restore_plan(
        source_job_id=source_job_id,
        source_status=source_status,
        weight_checkpoint=weight_checkpoint,
        expected_config_sha256=expected_config_sha256,
    )
    try:
        metadata_bytes = job_manager.read_artifact(
            source_job_id,
            TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
        ).payload
        weights_bytes = job_manager.read_artifact(
            source_job_id,
            TRAINING_WEIGHT_ARTIFACT_PATH,
        ).payload
    except KeyError:
        return {"error": "training_weight_artifact_not_found"}
    except (StudioJobArtifactUnavailable, ValueError):
        return {"error": "training_weight_artifact_unavailable"}

    fingerprint = training_architecture_fingerprint(config)
    record = job_manager.submit_process_task(
        kind="training",
        owner=STUDIO_TRAINING_WEIGHT_RESTORE_ATTACH_OWNER,
        request_id=None,
        task_path=TRAINING_ATTACH_PROCESS_TASK,
        payload={
            "config": config,
            "restore_plan": restore_plan.to_public_dict(),
            "architecture_fingerprint": fingerprint,
        },
        seed_inputs={
            TRAINING_ATTACH_SEED_WEIGHTS_PATH: weights_bytes,
            TRAINING_ATTACH_SEED_METADATA_PATH: metadata_bytes,
        },
    )
    proxy = TrainingJob(config, job_id=record.job_id)
    proxy.status = "running"
    _register_job(proxy)
    return {
        "job_id": record.job_id,
        "status": "running",
        "source_job_id": source_job_id,
        "architecture_fingerprint": fingerprint,
    }


def _request_live_training_weight_attach(
    target_job_id: str,
    source_job_id: str,
    job_manager: StudioJobManager,
    *,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Deliver verified source weights to a running training worker."""
    try:
        target_record = job_manager.record(target_job_id)
    except KeyError:
        return {"error": "training_job_not_found"}
    if target_record.status != "running" or target_record.execution_model != "process":
        return {"error": "training_job_not_running"}
    target_proxy, source_proxy = _get_registered_pair(target_job_id, source_job_id)
    target_config = dict(target_proxy.config) if target_proxy is not None else {}

    source_status = _get_training_status(source_job_id, job_manager)
    if "status" not in source_status:
        return {"error": "source_job_not_found"}
    weight_checkpoint = source_status.get("weight_checkpoint")
    if not isinstance(weight_checkpoint, dict):
        return {"error": "training_weight_checkpoint_unavailable"}
    source_job_status = cast(str, source_status["status"])

    restore_plan = build_training_weight_restore_plan(
        source_job_id=source_job_id,
        source_status=source_job_status,
        weight_checkpoint=weight_checkpoint,
        expected_config_sha256=expected_config_sha256,
    )
    try:
        metadata_bytes = job_manager.read_artifact(
            source_job_id,
            TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
        ).payload
        weights_bytes = job_manager.read_artifact(
            source_job_id,
            TRAINING_WEIGHT_ARTIFACT_PATH,
        ).payload
    except KeyError:
        return {"error": "training_weight_artifact_not_found"}
    except (StudioJobArtifactUnavailable, ValueError):
        return {"error": "training_weight_artifact_unavailable"}

    fingerprint = training_architecture_fingerprint(target_config)
    if source_proxy is not None and (
        training_architecture_fingerprint(source_proxy.config) != fingerprint
    ):
        return {"error": "architecture_incompatible"}

    try:
        job_manager.send_control_command(
            target_job_id,
            command={
                "action": "attach_weights",
                "restore_plan": restore_plan.to_public_dict(),
                "architecture_fingerprint": fingerprint,
                "weights_seed_path": _LIVE_ATTACH_WEIGHTS_SEED,
                "metadata_seed_path": _LIVE_ATTACH_METADATA_SEED,
            },
            seed_inputs={
                _LIVE_ATTACH_WEIGHTS_SEED: weights_bytes,
                _LIVE_ATTACH_METADATA_SEED: metadata_bytes,
            },
        )
    except StudioJobRejected:
        return {"error": "training_job_not_running"}
    return {
        "target_job_id": target_job_id,
        "source_job_id": source_job_id,
        "status": "attach_requested",
        "architecture_fingerprint": fingerprint,
    }
