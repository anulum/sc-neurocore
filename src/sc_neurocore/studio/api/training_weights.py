# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training-weight lifecycle routes

"""Restore and attach integrity-checked Studio training weights."""

from __future__ import annotations

import json
from typing import cast

from fastapi import APIRouter, HTTPException, Request

from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import (
    StudioTrainingWeightAttachRequest,
    StudioTrainingWeightLiveAttachRequest,
    StudioTrainingWeightRestoreRequest,
)
from sc_neurocore.studio.platform import (
    STUDIO_TRAINING_WEIGHT_RESTORE_OWNER,
    TRAINING_WEIGHT_ARTIFACT_PATH,
    TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
    TRAINING_WEIGHT_RESTORE_EVIDENCE_ARTIFACT_PATH,
    StudioJobArtifactUnavailable,
    StudioJobContext,
    build_training_weight_restore_evidence,
    build_training_weight_restore_plan,
    load_training_weight_state_dict,
    materialize_training_weight_payload,
)
from sc_neurocore.studio.training import (
    get_training_status,
    request_live_training_weight_attach,
    start_training_attach,
)


def build_training_weights_router(context: StudioApiContext) -> APIRouter:
    """Build the training-weight lifecycle router over shared Studio runtime state."""
    router = APIRouter()
    settings = context.settings
    studio_job_manager = context.studio_job_manager

    @router.post("/api/studio/training/weight-restore")
    def api_studio_training_weight_restore(
        restore_request: StudioTrainingWeightRestoreRequest,
        request: Request,
    ) -> dict[str, object]:
        """Materialize and verify a training job's weights as confined evidence.

        Builds the canonical restore plan from the source training job's stored
        checkpoint metadata, fetches the integrity-checked weight and metadata
        artifacts, then runs the untrusted torch deserialization inside a bounded
        worker job. The job emits a path-free
        ``studio.training.weight-restore.v1`` evidence artifact carrying only the
        verified digests and loaded-key totals; the in-memory tensor state
        dictionary never leaves the worker.
        """
        source_job_id = restore_request.source_job_id
        status_payload = get_training_status(source_job_id, studio_job_manager)
        if "status" not in status_payload:
            raise HTTPException(status_code=404, detail="training_job_not_found")
        weight_checkpoint = status_payload.get("weight_checkpoint")
        if not isinstance(weight_checkpoint, dict):
            raise HTTPException(
                status_code=409,
                detail="training_weight_checkpoint_unavailable",
            )
        source_status = status_payload.get("status")
        if not isinstance(source_status, str) or not source_status:
            raise HTTPException(status_code=409, detail="training_status_unavailable")
        try:
            restore_plan = build_training_weight_restore_plan(
                source_job_id=source_job_id,
                source_status=source_status,
                weight_checkpoint=weight_checkpoint,
                expected_config_sha256=restore_request.expected_config_sha256,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            metadata_payload = studio_job_manager.read_artifact(
                source_job_id,
                TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
            ).payload
            weights_payload = studio_job_manager.read_artifact(
                source_job_id,
                TRAINING_WEIGHT_ARTIFACT_PATH,
            ).payload
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="training_weight_artifact_not_found",
            ) from exc
        except (StudioJobArtifactUnavailable, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail="training_weight_artifact_unavailable",
            ) from exc

        restore_plan_payload = restore_plan.to_public_dict()
        request_id = getattr(request.state, "studio_request_id", None)

        def task(context: StudioJobContext) -> dict[str, object]:
            materialization = materialize_training_weight_payload(
                restore_plan=restore_plan_payload,
                metadata_payload=metadata_payload,
                weights_payload=weights_payload,
                trusted_loader=load_training_weight_state_dict,
            )
            evidence = build_training_weight_restore_evidence(
                materialization,
                source_status=source_status,
            )
            context.write_artifact(
                TRAINING_WEIGHT_RESTORE_EVIDENCE_ARTIFACT_PATH,
                json.dumps(evidence, indent=2, sort_keys=True),
            )
            return cast(dict[str, object], evidence)

        submitted = studio_job_manager.submit(
            kind="training",
            owner=STUDIO_TRAINING_WEIGHT_RESTORE_OWNER,
            request_id=request_id if isinstance(request_id, str) else None,
            task=task,
        )
        completed = studio_job_manager.wait(
            submitted.job_id,
            timeout_seconds=settings.job_default_timeout_seconds + 1.0,
        )
        if completed.status == "completed" and completed.result is not None:
            result = dict(completed.result)
            result["job_id"] = completed.job_id
            result["artifacts"] = [artifact.to_public_dict() for artifact in completed.artifacts]
            return result
        if completed.status in {"pending", "running", "cancelling"}:
            raise HTTPException(status_code=503, detail="studio_job_wait_exceeded")
        if completed.status == "timed_out":
            raise HTTPException(status_code=504, detail="studio_job_timed_out")
        raise HTTPException(status_code=500, detail="studio_job_failed")

    @router.post("/api/studio/training/weight-restore/attach")
    def api_studio_training_weight_restore_attach(
        attach_request: StudioTrainingWeightAttachRequest,
    ) -> dict[str, object]:
        """Warm-start a training job seeded with restored, verified weights.

        Builds the canonical restore plan from the source job's checkpoint,
        delivers the integrity-checked weights to a bounded process worker as
        confined seed inputs, and starts a training job that loads them at the
        epoch-zero checkpoint boundary before training forward. A strict load of
        an incompatible architecture fails the job before training begins. The
        worker writes a path-free ``studio.training.weight-restore-attach.v1``
        evidence artifact; the deserialized tensors never reach the response.
        """
        try:
            result = start_training_attach(
                attach_request.source_job_id,
                dict(attach_request.config),
                studio_job_manager,
                expected_config_sha256=attach_request.expected_config_sha256,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        error = result.get("error")
        if error == "training_job_not_found":
            raise HTTPException(status_code=404, detail=error)
        if error == "training_weight_artifact_not_found":
            raise HTTPException(status_code=404, detail=error)
        if error in {
            "training_weight_checkpoint_unavailable",
            "training_status_unavailable",
            "training_weight_artifact_unavailable",
        }:
            raise HTTPException(status_code=409, detail=error)
        if error is not None:
            raise HTTPException(status_code=500, detail="training_weight_attach_failed")
        return result

    @router.post("/api/studio/training/weight-restore/attach/live")
    def api_studio_training_weight_restore_attach_live(
        attach_request: StudioTrainingWeightLiveAttachRequest,
    ) -> dict[str, object]:
        """Deliver verified weights to a running training job for a live attach.

        Validates that the target job is running and that the source and target
        architectures are compatible, then delivers the integrity-checked weight
        artifacts to the running worker as a confined control command. The worker
        applies the attach at its next epoch boundary and writes a path-free
        ``studio.training.weight-restore-attach.v1`` (``mode: live``) evidence
        artifact. An incompatible attach is rejected without interrupting the
        running job. The response is returned immediately on delivery.
        """
        try:
            result = request_live_training_weight_attach(
                attach_request.target_job_id,
                attach_request.source_job_id,
                studio_job_manager,
                expected_config_sha256=attach_request.expected_config_sha256,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        error = result.get("error")
        if error in {"training_job_not_found", "source_job_not_found"}:
            raise HTTPException(status_code=404, detail=error)
        if error == "training_weight_artifact_not_found":
            raise HTTPException(status_code=404, detail=error)
        if error in {
            "training_job_not_running",
            "training_weight_checkpoint_unavailable",
            "training_status_unavailable",
            "training_weight_artifact_unavailable",
            "architecture_incompatible",
        }:
            raise HTTPException(status_code=409, detail=error)
        if error is not None:
            raise HTTPException(status_code=500, detail="training_weight_attach_failed")
        return result

    return router
