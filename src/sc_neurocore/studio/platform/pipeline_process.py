# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio pipeline process task

"""Importable process task for the Studio graph-to-synthesis pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from sc_neurocore.studio.platform.action_evidence import (
    write_studio_action_evidence_manifest,
)
from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.project import run_pipeline
from sc_neurocore.studio.synthesis import EdaProcessLimits, supported_targets

PIPELINE_PROCESS_TASK = "sc_neurocore.studio.platform.pipeline_process:run_pipeline_process_task"


def run_pipeline_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Run one Studio graph-to-synthesis pipeline in an isolated process.

    Parameters
    ----------
    context:
        Job sandbox used to write path-confined result and evidence artifacts.
    payload:
        JSON object matching the public ``/api/pipeline/run`` request contract.

    Returns
    -------
    dict[str, object]
        Path-free pipeline result payload.

    Raises
    ------
    ValueError
        If the payload does not match the JSON-serializable pipeline contract.
    """

    request = _pipeline_request_from_payload(payload)
    raw_result = run_pipeline(
        request.graph,
        request.target,
        process_limits=request.process_limits,
    )
    result = _result_mapping(raw_result)
    result_artifact = context.write_artifact(
        "pipeline/result.json",
        _serialize_worker_result(result),
    )
    write_studio_action_evidence_manifest(
        context,
        action_kind="studio.pipeline.run",
        result=result,
        result_artifact=result_artifact,
        evidence_artifact_path="pipeline/evidence.json",
        evidence_classification="compile",
        replay_route="POST /api/pipeline/run",
    )
    return result


class _PipelineProcessRequest:
    """Validated JSON payload for an isolated pipeline task."""

    def __init__(
        self,
        *,
        graph: dict[str, object],
        target: str,
        process_limits: EdaProcessLimits | None,
    ) -> None:
        self.graph = graph
        self.target = target
        self.process_limits = process_limits


def _pipeline_request_from_payload(payload: Mapping[str, object]) -> _PipelineProcessRequest:
    return _PipelineProcessRequest(
        graph=_graph_field(payload, "graph"),
        target=_target_field(payload, "target"),
        process_limits=_process_limits_from_payload(payload),
    )


def _graph_field(payload: Mapping[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Studio pipeline payload field {key!r} must be an object.")
    return cast(dict[str, object], value)


def _target_field(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key, "ice40")
    targets = supported_targets()
    if not isinstance(value, str) or value not in targets:
        raise ValueError(f"Studio pipeline payload field {key!r} must be one of {list(targets)!r}.")
    return value


def _process_limits_from_payload(payload: Mapping[str, object]) -> EdaProcessLimits | None:
    raw_cpu_seconds = payload.get("eda_process_cpu_seconds")
    raw_memory_bytes = payload.get("eda_process_memory_bytes")
    cpu_seconds = _optional_float_field(raw_cpu_seconds, "eda_process_cpu_seconds")
    memory_bytes = _optional_int_field(raw_memory_bytes, "eda_process_memory_bytes")
    if cpu_seconds is None and memory_bytes is None:
        return None
    return EdaProcessLimits(
        cpu_seconds=cpu_seconds,
        address_space_bytes=memory_bytes,
    )


def _optional_float_field(value: object, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Studio pipeline payload field {key!r} must be numeric or null.")
    return float(value)


def _optional_int_field(value: object, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Studio pipeline payload field {key!r} must be an integer or null.")
    return value


def _result_mapping(raw_result: object) -> dict[str, object]:
    if not isinstance(raw_result, dict):
        raise ValueError("Studio pipeline result must be a JSON object.")
    return cast(dict[str, object], raw_result)


def _serialize_worker_result(result: Mapping[str, object]) -> str:
    return json.dumps(dict(result), sort_keys=True, default=str)


__all__ = [
    "PIPELINE_PROCESS_TASK",
    "run_pipeline_process_task",
]
