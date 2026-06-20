# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio compiler process-task contracts

"""Contract tests for the importable Studio compiler process task."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.studio.platform.compile_process import run_compile_process_task
from sc_neurocore.studio.platform.jobs import StudioJobContext


def _context(tmp_path: Path) -> StudioJobContext:
    """Return a confined job context for direct process-task tests."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    return StudioJobContext(
        job_id="sj_compile_test",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )


def test_compile_process_task_writes_result_and_action_evidence(tmp_path: Path) -> None:
    """Compile task writes the same path-free result and evidence artifacts as the route."""

    context = _context(tmp_path)

    result = run_compile_process_task(
        context,
        {
            "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
            "init": None,
            "module_name": "sc_neuron",
            "params": {"C": 1.0, "E_L": -65.0, "tau_m": 10.0},
            "reset": "v = -65",
            "threshold": "v > -50",
        },
    )

    assert result["module_name"] == "sc_neuron"
    assert result["chars"] == len(cast(str, result["verilog"]))
    traceability = cast(dict[str, object], result["compile_traceability"])
    assert traceability["schema_version"] == "studio.compile-traceability.v1"
    assert traceability["evidence_classification"] == "compile"
    assert [artifact.relative_path for artifact in context.artifacts] == [
        "compiler/result.json",
        "compiler/evidence.json",
    ]
    result_payload = json.loads((tmp_path / "job" / "compiler" / "result.json").read_text())
    evidence_payload = json.loads((tmp_path / "job" / "compiler" / "evidence.json").read_text())
    assert result_payload == result
    assert evidence_payload["schema_version"] == "studio.action-evidence.v1"
    assert evidence_payload["action_kind"] == "studio.compile"
    assert evidence_payload["evidence_classification"] == "compile"
    assert evidence_payload["replay_route"] == "POST /api/compile"
    assert evidence_payload["job_id"] == "sj_compile_test"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"equations": []},
        {"equations": [1]},
        {"equations": ["dv/dt = -v"], "module_name": ""},
        {"equations": ["dv/dt = -v"], "params": {"tau": "slow"}},
    ],
)
def test_compile_process_task_rejects_invalid_payloads(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    """Invalid process payloads fail before compiler execution or artifact writes."""

    context = _context(tmp_path)

    with pytest.raises(ValueError, match="Studio compile payload"):
        run_compile_process_task(context, payload)

    assert context.artifacts == ()
