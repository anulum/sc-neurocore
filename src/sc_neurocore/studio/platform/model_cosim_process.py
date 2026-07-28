# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio selected-model co-simulation process task

"""Run selected-model bit-exact co-simulation inside an isolated Studio job."""

from __future__ import annotations

import json
from collections.abc import Mapping

from sc_neurocore.studio.model_compile_configuration import (
    resolve_model_compile_configuration,
)
from sc_neurocore.studio.model_cosim import run_model_cosim
from sc_neurocore.studio.platform.action_evidence import write_studio_action_evidence_manifest
from sc_neurocore.studio.platform.jobs import StudioJobContext

MODEL_COSIM_PROCESS_TASK = (
    "sc_neurocore.studio.platform.model_cosim_process:run_model_cosim_process_task"
)


def run_model_cosim_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Resolve one selected model and emit real external-tool parity evidence."""

    configuration = resolve_model_compile_configuration(payload)
    execution = run_model_cosim(
        configuration,
        current=_required_number(payload, "current"),
        n_steps=_required_step_count(payload),
    )
    context.write_artifact("cosim/model.v", execution.rtl_source.encode("utf-8"))
    context.write_artifact("cosim/testbench.v", execution.rtl_testbench.encode("utf-8"))
    context.write_artifact("cosim/reference.c", execution.reference_source.encode("utf-8"))
    context.write_artifact(
        "cosim/traces.json",
        json.dumps(
            {
                "reference": execution.reference_trace,
                "rtl": execution.rtl_trace,
                "signals": execution.report["signals"],
            },
            allow_nan=False,
            sort_keys=True,
        ).encode("utf-8"),
    )
    result_artifact = context.write_artifact(
        "cosim/report.json",
        json.dumps(execution.report, allow_nan=False, sort_keys=True).encode("utf-8"),
    )
    write_studio_action_evidence_manifest(
        context,
        action_kind="studio.models.cosim",
        result=execution.report,
        result_artifact=result_artifact,
        evidence_artifact_path="cosim/evidence.json",
        evidence_classification="cosim_parity",
        replay_route="POST /api/models/cosim",
    )
    return execution.report


def _required_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Studio model co-simulation field {key!r} must be a finite number.")
    numeric = float(value)
    if not (-float("inf") < numeric < float("inf")):
        raise ValueError(f"Studio model co-simulation field {key!r} must be a finite number.")
    return numeric


def _required_step_count(payload: Mapping[str, object]) -> int:
    value = payload.get("n_steps")
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2048:
        raise ValueError("Studio model co-simulation field 'n_steps' must be between 1 and 2048.")
    return value
