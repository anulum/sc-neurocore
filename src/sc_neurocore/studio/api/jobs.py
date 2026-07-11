# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job inspection routes

"""Expose bounded Studio job state and integrity-checked artifacts."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.responses import Response

from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.platform import StudioJobArtifactUnavailable


def build_jobs_router(context: StudioApiContext) -> APIRouter:
    """Build the job inspection router over shared Studio runtime state."""
    router = APIRouter()
    studio_job_manager = context.studio_job_manager

    @router.get("/api/studio/jobs/status")
    def api_studio_jobs_status() -> dict[str, object]:
        """Return path-free local worker health for operator dashboards."""
        return studio_job_manager.status().to_public_dict()

    @router.get("/api/studio/jobs")
    def api_studio_jobs() -> dict[str, object]:
        """Return path-free local job records for administrators."""
        return studio_job_manager.list_snapshot().to_public_dict()

    @router.get("/api/studio/jobs/{job_id}")
    def api_studio_job(job_id: str) -> dict[str, object]:
        """Return one path-free local job record for administrators."""
        try:
            return studio_job_manager.record(job_id).to_public_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_not_found") from exc

    @router.get("/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}")
    def api_studio_job_artifact(job_id: str, artifact_path: str) -> Response:
        """Download one declared Studio job artifact after integrity validation."""
        try:
            artifact_payload = studio_job_manager.read_artifact(job_id, artifact_path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_artifact_not_found") from exc
        except (StudioJobArtifactUnavailable, ValueError) as exc:
            raise HTTPException(status_code=409, detail="job_artifact_unavailable") from exc
        filename = Path(artifact_payload.artifact.relative_path).name or "artifact.bin"
        return Response(
            content=artifact_payload.payload,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Studio-Artifact-SHA256": artifact_payload.artifact.sha256,
                "X-Studio-Artifact-Size": str(artifact_payload.artifact.size_bytes),
            },
        )

    return router
