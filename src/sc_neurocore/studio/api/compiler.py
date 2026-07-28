# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio compiler routes

"""Adapt equation, NIR, and intermediate-representation compiler capabilities."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import (
    CompileRequest,
    ModelCompileRequest,
    ModelCosimRequest,
    SimulateRequest,
)
from sc_neurocore.studio.compiler import (
    build_ir_from_equation,
    emit_sv_from_equation,
    emit_systemverilog,
    verify_ir,
)
from sc_neurocore.studio.nir_compile import compile_nir_file_bytes
from sc_neurocore.studio.platform.compile_process import COMPILE_PROCESS_TASK
from sc_neurocore.studio.platform.model_compile_process import MODEL_COMPILE_PROCESS_TASK
from sc_neurocore.studio.platform.model_cosim_process import MODEL_COSIM_PROCESS_TASK


def build_compiler_router(context: StudioApiContext) -> APIRouter:
    """Build the compiler router over shared Studio runtime state."""
    router = APIRouter()
    run_studio_process_job_sync = context.run_studio_process_job_sync

    @router.post("/api/compile")
    def api_compile(req: CompileRequest) -> Any:
        return _safe(
            lambda: run_studio_process_job_sync(
                kind="compiler",
                owner="studio-compiler",
                task_path=COMPILE_PROCESS_TASK,
                payload=req.model_dump(),
            )
        )

    @router.post("/api/models/compile")
    def api_model_compile(req: ModelCompileRequest) -> Any:
        return _safe(
            lambda: run_studio_process_job_sync(
                kind="compiler",
                owner="studio-model-compiler",
                task_path=MODEL_COMPILE_PROCESS_TASK,
                payload=req.model_dump(),
            )
        )

    @router.post("/api/models/cosim")
    def api_model_cosim(req: ModelCosimRequest) -> Any:
        return _safe(
            lambda: run_studio_process_job_sync(
                kind="compiler",
                owner="studio-model-cosim",
                task_path=MODEL_COSIM_PROCESS_TASK,
                payload=req.model_dump(),
            )
        )

    @router.post("/api/nir/compile")
    async def api_nir_compile(
        request: Request,
        module_name: str = Query("sc_nir_network"),
        source_kind: str = Query("lfsr"),
    ) -> Any:
        data = await request.body()
        kind = source_kind if source_kind in ("lfsr", "sobol") else "lfsr"
        return _safe(
            lambda: compile_nir_file_bytes(data, module_name=module_name, source_kind=kind)
        )

    @router.post("/api/ir/build")
    def api_ir_build(req: SimulateRequest) -> Any:
        return _safe(
            lambda: build_ir_from_equation(
                equations=req.equations,
                params=req.params,
                threshold=req.threshold,
                reset=req.reset,
                dt=req.dt,
            )
        )

    @router.post("/api/ir/verify")
    def api_ir_verify(data: dict[str, Any]) -> Any:
        ir_text = data.get("ir_text", "")
        if not ir_text:
            raise HTTPException(422, "ir_text required")
        return _safe(lambda: verify_ir(ir_text))

    @router.post("/api/ir/emit-sv")
    def api_ir_emit_sv(data: dict[str, Any]) -> Any:
        ir_text = data.get("ir_text", "")
        if not ir_text:
            raise HTTPException(422, "ir_text required")
        return _safe(lambda: emit_systemverilog(ir_text))

    @router.post("/api/ir/emit-sv-direct")
    def api_ir_emit_sv_direct(req: SimulateRequest) -> Any:
        return _safe(
            lambda: emit_sv_from_equation(
                equations=req.equations,
                params=req.params,
                threshold=req.threshold,
                reset=req.reset,
            )
        )

    return router
