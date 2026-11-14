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
from pydantic import BaseModel

from sc_neurocore.neurons.equation_builder import from_equations
from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.compiler_result import decode_equation_ir_response
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import (
    CompileRequest,
    DirectCompileRequest,
    ModelCompileRequest,
    ModelCosimRequest,
)
from sc_neurocore.studio.compiler import (
    emit_sv_from_equation,
)
from sc_neurocore.studio.equation_ir import IROperation
from sc_neurocore.studio.equation_ir_schema import (
    EquationIREmitRequest,
    EquationIRRequest,
    EquationIRTextRequest,
)
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.model_compile_configuration import resolve_model_compile_configuration
from sc_neurocore.studio.nir_compile import compile_nir_file_bytes
from sc_neurocore.studio.platform.compile_process import COMPILE_PROCESS_TASK
from sc_neurocore.studio.platform.equation_ir_process import (
    EQUATION_IR_PROCESS_TASK,
)
from sc_neurocore.studio.platform.model_compile_process import MODEL_COMPILE_PROCESS_TASK
from sc_neurocore.studio.platform.model_cosim_process import MODEL_COSIM_PROCESS_TASK


def refuse_before_admission(payload: dict[str, Any]) -> None:
    """Answer a model compile the compiler would refuse with its reason, before any job runs.

    The worker resolves the same request and refuses it the same way, but a
    failed job is answered HTTP 500 "studio_job_failed" with its reason kept in
    the job ledger, so a caller could not tell a wrong parameter from a
    server fault. An authored refusal here is the caller's input: HTTP 422
    with its text. Anything else the resolver raises is a fault in the
    server's own compile configuration or code, not the caller's input. It is
    left to the job, which meets it again and records it as before; it must
    not reach the request boundary, which would answer a ``KeyError`` or a
    ``TypeError`` 422 "Invalid input", as if the caller had sent it.

    Parameters
    ----------
    payload : dict[str, Any]
        The request as the worker receives it.

    Raises
    ------
    HTTPException
        HTTP 422 carrying the refusal's text.
    """
    try:
        resolve_model_compile_configuration(payload, detail_getter=get_model_detail)
    except AuthoredRefusal as refusal:
        raise HTTPException(status_code=422, detail=str(refusal)) from None
    except Exception:
        return


def refuse_equations_before_admission(request: CompileRequest) -> None:
    """Answer an equation compile the parser would refuse with its reason, before any job runs.

    The worker's ``equation_to_fpga`` parses the same equations, threshold
    and reset before it emits RTL, and refuses them the same way, but a failed
    job is answered HTTP 500 "studio_job_failed" with its reason kept in the
    job ledger. This parse splits ODEs joined by ``;`` as that function does.
    An authored refusal here is the caller's input: HTTP 422 with its text.
    Anything else the parse raises is left to the job, as in
    :func:`refuse_before_admission`. Refusals the RTL emitter raises after the
    parse are not checked here.

    Parameters
    ----------
    request : CompileRequest
        The request as the route receives it.

    Raises
    ------
    HTTPException
        HTTP 422 carrying the refusal's text.
    """
    equations = [part.strip() for text in request.equations for part in text.split(";")]
    try:
        from_equations(
            *(equation for equation in equations if equation),
            threshold=request.threshold,
            reset=request.reset,
            params=request.params,
            init=request.init,
        )
    except AuthoredRefusal as refusal:
        raise HTTPException(status_code=422, detail=str(refusal)) from None
    except Exception:
        return


def build_compiler_router(context: StudioApiContext) -> APIRouter:
    """Build the compiler router over shared Studio runtime state."""
    router = APIRouter()
    run_studio_process_job_sync = context.run_studio_process_job_sync

    def run_equation_ir(operation: IROperation, request: BaseModel) -> dict[str, Any]:
        """Validate the bounded worker outcome and retain its exact job receipt."""
        response = run_studio_process_job_sync(
            kind="compiler",
            owner="studio-equation-ir",
            task_path=EQUATION_IR_PROCESS_TASK,
            payload={"operation": operation, "request": request.model_dump(mode="json")},
            include_job_receipt=True,
        )
        return decode_equation_ir_response(response)

    @router.post("/api/compile")
    def api_compile(req: CompileRequest) -> Any:
        refuse_equations_before_admission(req)
        return _safe(
            lambda: run_studio_process_job_sync(
                kind="compiler",
                owner="studio-compiler",
                task_path=COMPILE_PROCESS_TASK,
                payload=req.model_dump(),
            )
        )

    def run_model_job(owner: str, task_path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Refuse a request the compiler would refuse, then run the model job."""
        refuse_before_admission(payload)
        return run_studio_process_job_sync(
            kind="compiler", owner=owner, task_path=task_path, payload=payload
        )

    @router.post("/api/models/compile")
    def api_model_compile(req: ModelCompileRequest) -> Any:
        return _safe(
            lambda: run_model_job(
                "studio-model-compiler", MODEL_COMPILE_PROCESS_TASK, req.model_dump()
            )
        )

    @router.post("/api/models/cosim")
    def api_model_cosim(req: ModelCosimRequest) -> Any:
        return _safe(
            lambda: run_model_job("studio-model-cosim", MODEL_COSIM_PROCESS_TASK, req.model_dump())
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
    def api_ir_build(req: EquationIRRequest) -> Any:
        return _safe(lambda: run_equation_ir("build", req))

    @router.post("/api/ir/verify")
    def api_ir_verify(req: EquationIRTextRequest) -> Any:
        return _safe(lambda: run_equation_ir("verify", req))

    @router.post("/api/ir/replay")
    def api_ir_replay(req: EquationIRTextRequest) -> Any:
        return _safe(lambda: run_equation_ir("replay", req))

    @router.post("/api/ir/emit-sv")
    def api_ir_emit_sv(req: EquationIREmitRequest) -> Any:
        return _safe(lambda: run_equation_ir("emit-sv", req))

    @router.post("/api/ir/emit-sv-direct")
    def api_ir_emit_sv_direct(req: DirectCompileRequest) -> Any:
        return _safe(
            lambda: emit_sv_from_equation(
                equations=req.equations,
                params=req.params,
                threshold=req.threshold,
                reset=req.reset,
                init=req.init,
                module_name=req.module_name,
            )
        )

    return router
