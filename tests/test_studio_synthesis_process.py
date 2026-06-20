# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis process-task contracts

"""Contract tests for importable Studio synthesis process tasks."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.synthesis_process import (
    run_multi_target_synthesis_process_task,
    run_pnr_process_task,
    run_synthesis_process_task,
)


def _context(tmp_path: Path, job_id: str) -> StudioJobContext:
    """Return a confined job context for direct process-task tests."""

    work_dir = tmp_path / job_id
    work_dir.mkdir()
    return StudioJobContext(
        job_id=job_id,
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )


def _assert_evidence(
    tmp_path: Path,
    *,
    job_id: str,
    evidence_path: str,
    action_kind: str,
    replay_route: str,
) -> None:
    """Assert one process-task action evidence manifest."""

    evidence_payload = json.loads((tmp_path / job_id / evidence_path).read_text())
    assert evidence_payload["schema_version"] == "studio.action-evidence.v1"
    assert evidence_payload["action_kind"] == action_kind
    assert evidence_payload["evidence_classification"] == "synthesis"
    assert evidence_payload["job_id"] == job_id
    assert evidence_payload["replay_route"] == replay_route


def test_synthesis_process_task_writes_result_and_action_evidence(tmp_path: Path) -> None:
    """Synthesis task writes route-compatible result and evidence artifacts."""

    context = _context(tmp_path, "sj_synthesis_test")

    result = run_synthesis_process_task(
        context,
        {
            "eda_process_cpu_seconds": 30.0,
            "eda_process_memory_bytes": 268435456,
            "target": "ice40",
            "verilog": "module test(); endmodule",
        },
    )

    assert result["target"] == "ice40"
    assert "success" in result
    assert [artifact.relative_path for artifact in context.artifacts] == [
        "synthesis/result.json",
        "synthesis/evidence.json",
    ]
    assert json.loads((tmp_path / "sj_synthesis_test" / "synthesis" / "result.json").read_text()) == result
    _assert_evidence(
        tmp_path,
        job_id="sj_synthesis_test",
        evidence_path="synthesis/evidence.json",
        action_kind="studio.synthesis.run",
        replay_route="POST /api/synth/run",
    )


def test_multi_target_process_task_writes_result_and_action_evidence(tmp_path: Path) -> None:
    """Multi-target task writes route-compatible result and evidence artifacts."""

    context = _context(tmp_path, "sj_multi_target_test")

    result = run_multi_target_synthesis_process_task(
        context,
        {
            "eda_process_cpu_seconds": 30.0,
            "eda_process_memory_bytes": 268435456,
            "verilog": "module test(); endmodule",
        },
    )

    assert set(result) == {"supported", "target_provenance_matrix", "targets"}
    assert [artifact.relative_path for artifact in context.artifacts] == [
        "synthesis/multi-target-result.json",
        "synthesis/multi-target-evidence.json",
    ]
    assert (
        json.loads(
            (tmp_path / "sj_multi_target_test" / "synthesis" / "multi-target-result.json").read_text()
        )
        == result
    )
    _assert_evidence(
        tmp_path,
        job_id="sj_multi_target_test",
        evidence_path="synthesis/multi-target-evidence.json",
        action_kind="studio.synthesis.multi_target",
        replay_route="POST /api/synth/multi-target",
    )


def test_pnr_process_task_writes_result_and_action_evidence(tmp_path: Path) -> None:
    """PnR task writes route-compatible result and evidence artifacts."""

    netlist = tmp_path / "design.json"
    netlist.write_text("{}", encoding="utf-8")
    context = _context(tmp_path, "sj_pnr_test")

    result = run_pnr_process_task(
        context,
        {
            "eda_process_cpu_seconds": 30.0,
            "eda_process_memory_bytes": 268435456,
            "json_path": str(netlist),
            "target": "ice40",
        },
    )

    assert "success" in result
    assert [artifact.relative_path for artifact in context.artifacts] == [
        "synthesis/pnr-result.json",
        "synthesis/pnr-evidence.json",
    ]
    assert json.loads((tmp_path / "sj_pnr_test" / "synthesis" / "pnr-result.json").read_text()) == result
    _assert_evidence(
        tmp_path,
        job_id="sj_pnr_test",
        evidence_path="synthesis/pnr-evidence.json",
        action_kind="studio.synthesis.pnr",
        replay_route="POST /api/synth/pnr",
    )


@pytest.mark.parametrize(
    ("task_name", "payload"),
    [
        ("synthesis", {}),
        ("synthesis", {"verilog": "", "target": "ice40"}),
        ("synthesis", {"verilog": "module t(); endmodule", "target": "unknown"}),
        ("synthesis", {"verilog": "module t(); endmodule", "eda_process_cpu_seconds": "fast"}),
        ("multi", {}),
        ("multi", {"verilog": "module t(); endmodule", "eda_process_memory_bytes": 1.5}),
        ("pnr", {"target": "ice40"}),
        ("pnr", {"json_path": "", "target": "ice40"}),
        ("pnr", {"json_path": "design.json", "target": "unknown"}),
    ],
)
def test_synthesis_process_tasks_reject_invalid_payloads(
    tmp_path: Path,
    task_name: str,
    payload: dict[str, object],
) -> None:
    """Invalid process payloads fail before synthesis artifacts are written."""

    context = _context(tmp_path, f"sj_invalid_{task_name}")
    tasks = {
        "multi": run_multi_target_synthesis_process_task,
        "pnr": run_pnr_process_task,
        "synthesis": run_synthesis_process_task,
    }

    with pytest.raises(ValueError, match="Studio synthesis payload"):
        tasks[task_name](context, payload)

    assert context.artifacts == ()
