# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio synthesis and place-and-route routes

"""Run bounded synthesis, estimation, and place-and-route adapters."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.platform.synthesis_process import (
    SYNTHESIS_MULTI_TARGET_PROCESS_TASK,
    SYNTHESIS_PNR_PROCESS_TASK,
    SYNTHESIS_RUN_PROCESS_TASK,
)
from sc_neurocore.studio.synthesis import (
    check_tools,
    estimate_resources,
    supported_targets,
)


def _validate_synthesis_verilog(verilog: Any) -> str:
    """Return validated Verilog source for bounded synthesis worker routes."""
    if not isinstance(verilog, str):
        raise HTTPException(422, "verilog source must be a string")
    if not verilog.strip():
        raise HTTPException(422, "verilog source required")
    if len(verilog.encode("utf-8")) > 2 * 1024 * 1024:
        raise HTTPException(422, "verilog source exceeds 2 MiB size limit")
    return verilog


def _validate_synthesis_target(target: Any) -> str:
    """Return a validated synthesis target identifier."""
    targets = supported_targets()
    if not isinstance(target, str) or target not in targets:
        raise HTTPException(422, f"unknown synthesis target; supported targets: {list(targets)}")
    return target


def build_synthesis_router(context: StudioApiContext) -> APIRouter:
    """Build the synthesis and place-and-route router over shared Studio runtime state."""
    router = APIRouter()
    eda_process_limits = context.eda_process_limits
    run_studio_process_job_sync = context.run_studio_process_job_sync

    @router.get("/api/synth/tools-status")
    def api_synth_tools() -> Any:
        return check_tools()

    @router.post("/api/synth/run")
    def api_synth_run(data: dict[str, Any]) -> Any:
        verilog = _validate_synthesis_verilog(data.get("verilog", ""))
        target = _validate_synthesis_target(data.get("target", "ice40"))
        return _safe(
            lambda: run_studio_process_job_sync(
                kind="synthesis",
                owner="studio-synthesis",
                task_path=SYNTHESIS_RUN_PROCESS_TASK,
                payload={
                    "eda_process_cpu_seconds": eda_process_limits.cpu_seconds,
                    "eda_process_memory_bytes": eda_process_limits.address_space_bytes,
                    "target": target,
                    "verilog": verilog,
                },
            )
        )

    @router.post("/api/synth/multi-target")
    def api_synth_multi(data: dict[str, Any]) -> Any:
        verilog = _validate_synthesis_verilog(data.get("verilog", ""))
        return _safe(
            lambda: run_studio_process_job_sync(
                kind="synthesis",
                owner="studio-synthesis",
                task_path=SYNTHESIS_MULTI_TARGET_PROCESS_TASK,
                payload={
                    "eda_process_cpu_seconds": eda_process_limits.cpu_seconds,
                    "eda_process_memory_bytes": eda_process_limits.address_space_bytes,
                    "verilog": verilog,
                },
            )
        )

    @router.post("/api/synth/estimate")
    def api_synth_estimate(data: dict[str, Any]) -> Any:
        raw_ir_op_count = data.get("ir_op_count", 0)
        target = data.get("target", "ice40")
        if not isinstance(raw_ir_op_count, int):
            raise HTTPException(422, "ir_op_count must be an integer >= 1")
        ir_op_count = raw_ir_op_count
        if ir_op_count < 1:
            raise HTTPException(422, "ir_op_count must be >= 1")
        return _safe(lambda: estimate_resources(ir_op_count, target))

    @router.post("/api/synth/pnr")
    def api_synth_pnr(data: dict[str, Any]) -> Any:
        json_path = data.get("json_path", "")
        if not json_path:
            raise HTTPException(422, "json_path required")
        target = _validate_synthesis_target(data.get("target", "ice40"))
        return _safe(
            lambda: run_studio_process_job_sync(
                kind="synthesis",
                owner="studio-pnr",
                task_path=SYNTHESIS_PNR_PROCESS_TASK,
                payload={
                    "eda_process_cpu_seconds": eda_process_limits.cpu_seconds,
                    "eda_process_memory_bytes": eda_process_limits.address_space_bytes,
                    "json_path": json_path,
                    "target": target,
                },
            )
        )

    return router
