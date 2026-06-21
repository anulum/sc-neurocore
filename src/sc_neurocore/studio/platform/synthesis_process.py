# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis process tasks

"""Importable process tasks for Studio synthesis and PnR endpoints."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from sc_neurocore.studio.platform.action_evidence import (
    write_studio_action_evidence_manifest,
)
from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.synthesis import (
    EdaProcessLimits,
    multi_target_synthesis,
    run_pnr,
    run_synthesis,
    supported_targets,
)

SYNTHESIS_RUN_PROCESS_TASK = (
    "sc_neurocore.studio.platform.synthesis_process:run_synthesis_process_task"
)
SYNTHESIS_MULTI_TARGET_PROCESS_TASK = (
    "sc_neurocore.studio.platform.synthesis_process:run_multi_target_synthesis_process_task"
)
SYNTHESIS_PNR_PROCESS_TASK = "sc_neurocore.studio.platform.synthesis_process:run_pnr_process_task"


def run_synthesis_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Run one Studio synthesis request in an isolated process.

    Parameters
    ----------
    context:
        Job sandbox used to write path-confined result and evidence artifacts.
    payload:
        JSON object matching the public ``/api/synth/run`` request contract.

    Returns
    -------
    dict[str, object]
        Path-free synthesis result payload.

    Raises
    ------
    ValueError
        If the payload does not match the JSON-serializable synthesis contract.
    """

    request = _synthesis_request_from_payload(payload)
    return _run_with_evidence(
        context,
        action_kind="studio.synthesis.run",
        evidence_artifact_path="synthesis/evidence.json",
        result_artifact_path="synthesis/result.json",
        replay_route="POST /api/synth/run",
        task=lambda: run_synthesis(
            request.verilog,
            request.target,
            process_limits=request.process_limits,
        ),
    )


def run_multi_target_synthesis_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Run one Studio multi-target synthesis request in an isolated process.

    Parameters
    ----------
    context:
        Job sandbox used to write path-confined result and evidence artifacts.
    payload:
        JSON object matching the public ``/api/synth/multi-target`` contract.

    Returns
    -------
    dict[str, object]
        Path-free multi-target synthesis result payload.

    Raises
    ------
    ValueError
        If the payload does not match the JSON-serializable synthesis contract.
    """

    request = _multi_target_request_from_payload(payload)
    return _run_with_evidence(
        context,
        action_kind="studio.synthesis.multi_target",
        evidence_artifact_path="synthesis/multi-target-evidence.json",
        result_artifact_path="synthesis/multi-target-result.json",
        replay_route="POST /api/synth/multi-target",
        task=lambda: multi_target_synthesis(
            request.verilog,
            process_limits=request.process_limits,
        ),
    )


def run_pnr_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Run one Studio PnR request in an isolated process.

    Parameters
    ----------
    context:
        Job sandbox used to write path-confined result and evidence artifacts.
    payload:
        JSON object matching the public ``/api/synth/pnr`` request contract.

    Returns
    -------
    dict[str, object]
        Path-free PnR result payload.

    Raises
    ------
    ValueError
        If the payload does not match the JSON-serializable PnR contract.
    """

    request = _pnr_request_from_payload(payload)
    return _run_with_evidence(
        context,
        action_kind="studio.synthesis.pnr",
        evidence_artifact_path="synthesis/pnr-evidence.json",
        result_artifact_path="synthesis/pnr-result.json",
        replay_route="POST /api/synth/pnr",
        task=lambda: run_pnr(
            request.json_path,
            request.target,
            process_limits=request.process_limits,
        ),
    )


class _SynthesisProcessRequest:
    """Validated JSON payload for isolated synthesis tasks."""

    def __init__(
        self,
        *,
        verilog: str,
        target: str,
        process_limits: EdaProcessLimits | None,
    ) -> None:
        self.verilog = verilog
        self.target = target
        self.process_limits = process_limits


class _MultiTargetSynthesisProcessRequest:
    """Validated JSON payload for isolated multi-target synthesis tasks."""

    def __init__(self, *, verilog: str, process_limits: EdaProcessLimits | None) -> None:
        self.verilog = verilog
        self.process_limits = process_limits


class _PnrProcessRequest:
    """Validated JSON payload for an isolated PnR task."""

    def __init__(
        self,
        *,
        json_path: str,
        target: str,
        process_limits: EdaProcessLimits | None,
    ) -> None:
        self.json_path = json_path
        self.target = target
        self.process_limits = process_limits


def _synthesis_request_from_payload(payload: Mapping[str, object]) -> _SynthesisProcessRequest:
    return _SynthesisProcessRequest(
        verilog=_verilog_field(payload, "verilog"),
        target=_target_field(payload, "target"),
        process_limits=_process_limits_from_payload(payload),
    )


def _multi_target_request_from_payload(
    payload: Mapping[str, object],
) -> _MultiTargetSynthesisProcessRequest:
    return _MultiTargetSynthesisProcessRequest(
        verilog=_verilog_field(payload, "verilog"),
        process_limits=_process_limits_from_payload(payload),
    )


def _pnr_request_from_payload(payload: Mapping[str, object]) -> _PnrProcessRequest:
    return _PnrProcessRequest(
        json_path=_string_field(payload, "json_path"),
        target=_target_field(payload, "target"),
        process_limits=_process_limits_from_payload(payload),
    )


def _verilog_field(payload: Mapping[str, object], key: str) -> str:
    value = _string_field(payload, key)
    if len(value.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValueError(f"Studio synthesis payload field {key!r} exceeds 2 MiB.")
    return value


def _string_field(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Studio synthesis payload field {key!r} must be a non-empty string.")
    return value


def _target_field(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key, "ice40")
    targets = supported_targets()
    if not isinstance(value, str) or value not in targets:
        raise ValueError(
            f"Studio synthesis payload field {key!r} must be one of {list(targets)!r}."
        )
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
        raise ValueError(f"Studio synthesis payload field {key!r} must be numeric or null.")
    return float(value)


def _optional_int_field(value: object, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Studio synthesis payload field {key!r} must be an integer or null.")
    return value


def _run_with_evidence(
    context: StudioJobContext,
    *,
    action_kind: str,
    evidence_artifact_path: str,
    result_artifact_path: str,
    replay_route: str,
    task: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    result = dict(task())
    result_artifact = context.write_artifact(
        result_artifact_path,
        _serialize_worker_result(result),
    )
    write_studio_action_evidence_manifest(
        context,
        action_kind=action_kind,
        result=result,
        result_artifact=result_artifact,
        evidence_artifact_path=evidence_artifact_path,
        evidence_classification="synthesis",
        replay_route=replay_route,
    )
    return result


def _serialize_worker_result(result: Mapping[str, object]) -> str:
    return json.dumps(dict(result), sort_keys=True, default=str)


__all__ = [
    "SYNTHESIS_MULTI_TARGET_PROCESS_TASK",
    "SYNTHESIS_PNR_PROCESS_TASK",
    "SYNTHESIS_RUN_PROCESS_TASK",
    "run_multi_target_synthesis_process_task",
    "run_pnr_process_task",
    "run_synthesis_process_task",
]
