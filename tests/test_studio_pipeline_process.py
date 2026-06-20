# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio pipeline process-task contracts

"""Contract tests for the importable Studio pipeline process task."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.pipeline_process import run_pipeline_process_task


def _context(tmp_path: Path) -> StudioJobContext:
    """Return a confined job context for direct process-task tests."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    return StudioJobContext(
        job_id="sj_pipeline_test",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )


def test_pipeline_process_task_writes_result_and_action_evidence(tmp_path: Path) -> None:
    """Pipeline task writes the route-compatible result and evidence artifacts."""

    context = _context(tmp_path)

    result = run_pipeline_process_task(
        context,
        {
            "eda_process_cpu_seconds": 30.0,
            "eda_process_memory_bytes": 268435456,
            "graph": {"populations": [], "projections": []},
            "target": "ice40",
        },
    )

    assert result == {
        "errors": ["Network has no populations"],
        "step": "validate",
        "success": False,
    }
    assert [artifact.relative_path for artifact in context.artifacts] == [
        "pipeline/result.json",
        "pipeline/evidence.json",
    ]
    result_payload = json.loads((tmp_path / "job" / "pipeline" / "result.json").read_text())
    evidence_payload = json.loads((tmp_path / "job" / "pipeline" / "evidence.json").read_text())
    assert result_payload == result
    assert evidence_payload["schema_version"] == "studio.action-evidence.v1"
    assert evidence_payload["action_kind"] == "studio.pipeline.run"
    assert evidence_payload["evidence_classification"] == "compile"
    assert evidence_payload["replay_route"] == "POST /api/pipeline/run"
    assert evidence_payload["job_id"] == "sj_pipeline_test"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"graph": []},
        {"graph": {}, "target": "unknown"},
        {"graph": {}, "target": "ice40", "eda_process_cpu_seconds": "fast"},
        {"graph": {}, "target": "ice40", "eda_process_memory_bytes": 1.5},
        {"graph": {}, "target": "ice40", "eda_process_cpu_seconds": 0},
    ],
)
def test_pipeline_process_task_rejects_invalid_payloads(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    """Invalid process payloads fail before pipeline artifacts are written."""

    context = _context(tmp_path)

    with pytest.raises(ValueError, match="Studio pipeline payload|EDA process CPU limit"):
        run_pipeline_process_task(context, payload)

    assert context.artifacts == ()
