# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training-monitor routes

"""Start, inspect, stop, checkpoint, and stream Studio training jobs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.training import (
    export_training_checkpoint,
    get_training_status,
    import_training_checkpoint,
    list_cell_types,
    list_jobs,
    list_surrogates,
    start_training,
    stop_training,
    stream_metrics,
)


def build_training_router(context: StudioApiContext) -> APIRouter:
    """Build the training monitor router over shared Studio runtime state."""
    router = APIRouter()
    studio_job_manager = context.studio_job_manager

    @router.get("/api/training/surrogates")
    def api_surrogates() -> Any:
        return list_surrogates()

    @router.get("/api/training/cell-types")
    def api_cell_types() -> Any:
        return list_cell_types()

    @router.post("/api/training/start")
    def api_training_start(data: dict[str, Any]) -> Any:
        return _safe(lambda: start_training(data, studio_job_manager))

    @router.post("/api/training/stop")
    def api_training_stop(data: dict[str, Any]) -> Any:
        job_id = data.get("job_id", "")
        if not job_id:
            raise HTTPException(422, "job_id required")
        return stop_training(job_id, studio_job_manager)

    @router.get("/api/training/jobs")
    def api_training_jobs() -> Any:
        return list_jobs()

    @router.get("/api/training/status/{job_id}")
    def api_training_status(job_id: str) -> Any:
        result = get_training_status(job_id, studio_job_manager)
        if result.get("error") and "job_id" not in result:
            raise HTTPException(404, result["error"])
        return result

    @router.get("/api/training/checkpoint/{job_id}")
    def api_training_checkpoint_export(job_id: str) -> Any:
        """Export one portable Training Monitor checkpoint."""
        result = export_training_checkpoint(job_id, studio_job_manager)
        if result.get("error"):
            raise HTTPException(404, result["error"])
        return result

    @router.post("/api/training/checkpoint/import")
    def api_training_checkpoint_import(data: dict[str, Any]) -> Any:
        """Validate an imported Training Monitor checkpoint."""
        return _safe(lambda: import_training_checkpoint(data))

    @router.get("/api/training/stream/{job_id}")
    def api_training_stream(job_id: str) -> Any:
        from starlette.responses import StreamingResponse

        return StreamingResponse(
            stream_metrics(job_id, studio_job_manager),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
