# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — hosted scientific process adapters

"""Publish actual existing worker outcomes as finite digest-custodied results."""

from __future__ import annotations

import json
from collections.abc import Mapping

from sc_neurocore.studio.api.analysis_jobs import (
    run_analysis_job_task,
    validate_analysis_job_request,
)
from sc_neurocore.studio.api.schemas import AnalysisJobRequest
from sc_neurocore.studio.api.host_jobs_schema import (
    HostAnalysisRequest,
    HostJobRequest,
    parse_host_job,
)
from sc_neurocore.studio.platform.equation_ir_process import (
    EquationIRWorkerResult,
    EquationIRWorkerSuccess,
    run_equation_ir_process_task,
)
from sc_neurocore.studio.platform.jobs import StudioJobContext

HOST_ANALYSIS_TASK = "sc_neurocore.studio.api.host_job_tasks:run_host_analysis_task"
HOST_COMPILER_TASK = "sc_neurocore.studio.api.host_job_tasks:run_host_compiler_task"


def _request(payload: Mapping[str, object]) -> HostJobRequest:
    """Revalidate original text after sorted ledger/process transport."""
    text = payload.get("envelope_json")
    if set(payload) != {"envelope_json"} or not isinstance(text, str):
        raise ValueError("Invalid hosted worker envelope")
    return parse_host_job(text)


def _publish(context: StudioJobContext, result: dict[str, object]) -> dict[str, object]:
    """Seal only a real outcome after the task's cancellation checkpoint."""
    context.check_cancelled()
    context.write_artifact("host/result.json", json.dumps(result, allow_nan=False))
    return result


def run_host_analysis_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Run a validated existing analysis in the admitted process sandbox.

    Parameters
    ----------
    context : StudioJobContext
        Actual supervised worker's cancellation and artifact custody.
    payload : Mapping of str to object
        Original envelope text; scientific dictionary order survives transport.

    Returns
    -------
    dict of str to object
        Existing scientific result, also sealed as ``host/result.json``.

    Raises
    ------
    ValueError
        The envelope selects a different task family or invalid input.
    StudioJobCancelled
        Cancellation was requested before execution or publication.
    """
    context.check_cancelled()
    request = _request(payload)
    if not isinstance(request, HostAnalysisRequest):
        raise ValueError("Hosted task requires an analysis operation")
    analysis, normalized, _, _, _ = validate_analysis_job_request(
        AnalysisJobRequest(analysis=request.operation, payload=request.request)
    )
    return _publish(context, run_analysis_job_task(analysis, normalized, context))


def run_host_compiler_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Run the native compiler without confusing transport completion with acceptance.

    Parameters
    ----------
    context : StudioJobContext
        Actual supervised worker's cancellation and artifact custody.
    payload : Mapping of str to object
        Original closed compiler envelope text, preserving exact source inputs.

    Returns
    -------
    dict of str to object
        Existing accepted/refused compiler envelope and its actual evidence.

    Raises
    ------
    ValueError
        The envelope is invalid or selects an analysis task.
    StudioJobCancelled
        Cancellation was requested before execution or publication.
    """
    context.check_cancelled()
    request = _request(payload)
    if isinstance(request, HostAnalysisRequest):
        raise ValueError("Hosted task requires a compiler operation")
    outcome = run_equation_ir_process_task(
        context,
        {
            "operation": request.operation.removeprefix("ir."),
            "request": request.request.model_dump(mode="json"),
        },
    )
    validated = EquationIRWorkerResult.model_validate(outcome).root
    if isinstance(validated, EquationIRWorkerSuccess):
        context.check_cancelled()
        context.write_artifact("host/graph.scir", validated.result.ir_text)
        if validated.result.systemverilog is not None:
            context.write_artifact("host/output.sv", validated.result.systemverilog)
    return _publish(context, outcome)
