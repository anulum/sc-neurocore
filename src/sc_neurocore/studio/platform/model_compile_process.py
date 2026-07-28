# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio catalogue-model compiler process task

"""Compile one canonical catalogue schema in an isolated Studio job."""

from __future__ import annotations

import json
from collections.abc import Mapping

from sc_neurocore.studio.compile_traceability import build_model_compile_traceability
from sc_neurocore.studio.model_compile_configuration import (
    resolve_model_compile_configuration,
)
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.platform.action_evidence import write_studio_action_evidence_manifest
from sc_neurocore.studio.platform.jobs import StudioJobContext

MODEL_COMPILE_PROCESS_TASK = (
    "sc_neurocore.studio.platform.model_compile_process:run_model_compile_process_task"
)


def run_model_compile_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Resolve and compile one catalogue model through its canonical schema."""

    configuration = resolve_model_compile_configuration(payload, detail_getter=get_model_detail)
    verilog = configuration.to_verilog()
    result: dict[str, object] = {
        "chars": len(verilog),
        "compile_configuration": configuration.to_public_dict(),
        "compile_traceability": build_model_compile_traceability(
            model_name=configuration.model_name,
            schema_name=configuration.schema_name,
            schema_sha256=configuration.schema_sha256,
            params=configuration.params,
            dt=configuration.dt,
            integrator=configuration.integrator,
            q_format=configuration.q_format.q_label,
            module_name=configuration.module_name,
            verilog=verilog,
        ).to_public_dict(),
        "module_name": configuration.module_name,
        "verilog": verilog,
    }
    result_artifact = context.write_artifact(
        "compiler/model-result.json",
        json.dumps(result, allow_nan=False, sort_keys=True).encode("utf-8"),
    )
    write_studio_action_evidence_manifest(
        context,
        action_kind="studio.models.compile",
        result=result,
        result_artifact=result_artifact,
        evidence_artifact_path="compiler/model-evidence.json",
        evidence_classification="compile",
        replay_route="POST /api/models/compile",
    )
    return result
