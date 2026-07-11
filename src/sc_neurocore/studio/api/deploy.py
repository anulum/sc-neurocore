# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio deployment-pipeline routes

"""Run the bounded graph-to-target Studio deployment pipeline."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.synthesis import _validate_synthesis_target
from sc_neurocore.studio.platform.pipeline_process import PIPELINE_PROCESS_TASK


def build_deploy_router(context: StudioApiContext) -> APIRouter:
    """Build the deployment pipeline router over shared Studio runtime state."""
    router = APIRouter()
    eda_process_limits = context.eda_process_limits
    run_studio_process_job_sync = context.run_studio_process_job_sync

    @router.post("/api/pipeline/run")
    def api_pipeline_run(data: dict[str, Any]) -> Any:
        graph = data.get("graph", {})
        target = _validate_synthesis_target(data.get("target", "ice40"))
        return _safe(
            lambda: run_studio_process_job_sync(
                kind="compiler",
                owner="studio-pipeline",
                task_path=PIPELINE_PROCESS_TASK,
                payload={
                    "eda_process_cpu_seconds": eda_process_limits.cpu_seconds,
                    "eda_process_memory_bytes": eda_process_limits.address_space_bytes,
                    "graph": graph,
                    "target": target,
                },
            )
        )

    return router
