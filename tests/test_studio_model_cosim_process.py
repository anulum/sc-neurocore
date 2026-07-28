# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio selected-model co-simulation process contracts

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.model_cosim_process import run_model_cosim_process_task

HAS_COSIM_TOOLS = all(shutil.which(tool) is not None for tool in ("gcc", "iverilog", "vvp"))


def _context(tmp_path: Path) -> StudioJobContext:
    work_dir = tmp_path / "job"
    work_dir.mkdir()
    return StudioJobContext(
        job_id="sj_model_cosim_test",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=2 * 1024 * 1024,
    )


@pytest.mark.skipif(not HAS_COSIM_TOOLS, reason="GCC and Icarus Verilog are required")
def test_model_cosim_process_writes_report_traces_sources_and_evidence(tmp_path: Path) -> None:
    context = _context(tmp_path)

    result = run_model_cosim_process_task(
        context,
        {
            "model_name": "AdaptiveThresholdIFNeuron",
            "integrator": "map",
            "q_format": "Q8.8",
            "current": 10.0,
            "n_steps": 12,
        },
    )

    assert result["bit_exact"] is True
    assert [artifact.relative_path for artifact in context.artifacts] == [
        "cosim/model.v",
        "cosim/testbench.v",
        "cosim/reference.c",
        "cosim/traces.json",
        "cosim/report.json",
        "cosim/evidence.json",
    ]
    traces = json.loads((tmp_path / "job/cosim/traces.json").read_text())
    assert len(traces["rtl"]) == len(traces["reference"]) == 12
    evidence = json.loads((tmp_path / "job/cosim/evidence.json").read_text())
    assert evidence["action_kind"] == "studio.models.cosim"
    assert evidence["evidence_classification"] == "cosim_parity"
    assert evidence["replay_route"] == "POST /api/models/cosim"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"model_name": "AdaptiveThresholdIFNeuron", "current": "high", "n_steps": 4}, "current"),
        (
            {"model_name": "AdaptiveThresholdIFNeuron", "current": float("nan"), "n_steps": 4},
            "current",
        ),
        ({"model_name": "AdaptiveThresholdIFNeuron", "current": 1.0, "n_steps": 0}, "n_steps"),
        ({"model_name": "LapicqueNeuron", "current": 1.0, "n_steps": 4}, "supports integrators"),
    ],
)
def test_model_cosim_process_rejects_invalid_or_unsupported_payload(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    context = _context(tmp_path)

    with pytest.raises((RuntimeError, ValueError), match=message):
        run_model_cosim_process_task(context, payload)

    assert context.artifacts == ()
