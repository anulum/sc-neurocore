# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio catalogue-model compiler process contracts

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.studio.platform import model_compile_process
from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.model_compile_process import run_model_compile_process_task


def _context(tmp_path: Path) -> StudioJobContext:
    work_dir = tmp_path / "job"
    work_dir.mkdir()
    return StudioJobContext(
        job_id="sj_model_compile_test",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=2 * 1024 * 1024,
    )


def test_model_compile_process_writes_rtl_and_action_evidence(tmp_path: Path) -> None:
    context = _context(tmp_path)

    result = run_model_compile_process_task(
        context,
        {
            "model_name": "LapicqueNeuron",
            "params": {"tau": 15.0},
            "dt": 1.0,
            "integrator": "exp_euler",
            "q_format": "Q16.16",
            "module_name": "sc_lapicque_selected",
        },
    )

    assert "module sc_lapicque_selected" in cast(str, result["verilog"])
    assert result["chars"] == len(cast(str, result["verilog"]))
    traceability = cast(dict[str, object], result["compile_traceability"])
    source_payload = cast(dict[str, object], traceability["source_payload"])
    assert result["compile_configuration"] == {
        "dt": 1.0,
        "integrator": "exp_euler",
        "model_name": "LapicqueNeuron",
        "q_format": "Q16.16",
        "schema_name": "lapicque",
        "schema_sha256": source_payload["schema_sha256"],
    }
    assert [artifact.relative_path for artifact in context.artifacts] == [
        "compiler/model-result.json",
        "compiler/model-evidence.json",
    ]
    evidence = json.loads((tmp_path / "job/compiler/model-evidence.json").read_text())
    assert evidence["action_kind"] == "studio.models.compile"
    assert evidence["replay_route"] == "POST /api/models/compile"


def test_model_compile_process_uses_schema_defaults(tmp_path: Path) -> None:
    context = _context(tmp_path)

    result = run_model_compile_process_task(
        context,
        {"model_name": "LapicqueNeuron", "q_format": "Q8.8"},
    )

    assert result["module_name"] == "sc_lapicque_neuron"
    configuration = cast(dict[str, object], result["compile_configuration"])
    assert configuration["dt"] == 1.0
    assert configuration["integrator"] == "exp_euler"
    assert configuration["schema_name"] == "lapicque"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "model_name"),
        ({"model_name": "MissingNeuron"}, "Unknown Studio model"),
        ({"model_name": "ATypeKNeuron"}, "no canonical schema"),
        ({"model_name": "LapicqueNeuron", "integrator": "rk4"}, "not declared"),
        ({"model_name": "LapicqueNeuron", "params": {"missing": 1}}, "Unknown schema"),
        ({"model_name": "LapicqueNeuron", "q_format": "Q1.0"}, "between 2 and 64"),
        ({"model_name": "LapicqueNeuron", "integrator": 1}, "integrator"),
        ({"model_name": "LapicqueNeuron", "params": []}, "params"),
        ({"model_name": "LapicqueNeuron", "params": {"tau": True}}, "finite numbers"),
        ({"model_name": "LapicqueNeuron", "params": {"tau": float("nan")}}, "finite numbers"),
        ({"model_name": "LapicqueNeuron", "dt": "slow"}, "positive number"),
        ({"model_name": "LapicqueNeuron", "dt": 0}, "positive number"),
    ],
)
def test_model_compile_process_rejects_invalid_payloads(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    context = _context(tmp_path)

    with pytest.raises(ValueError, match=message):
        run_model_compile_process_task(context, payload)

    assert context.artifacts == ()


def test_model_compile_process_rejects_corrupt_catalogue_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(
        model_compile_process,
        "get_model_detail",
        lambda _name: {"compile_configuration": {"schema_name": "lapicque", "integrators": []}},
    )

    with pytest.raises(ValueError, match="configuration 'integrators' is invalid"):
        run_model_compile_process_task(context, {"model_name": "LapicqueNeuron"})

    assert context.artifacts == ()


@pytest.mark.parametrize(
    "schema, message",
    [
        ({"integration": [], "parameters": {}}, "invalid integration"),
        (
            {"integration": {"method": "exp_euler", "dt": 1.0}, "parameters": []},
            "invalid parameters",
        ),
    ],
)
def test_model_compile_process_rejects_corrupt_schema_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema: dict[str, object],
    message: str,
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(
        "sc_neurocore.studio.model_compile_configuration.load_schema",
        lambda _name: schema,
    )

    with pytest.raises(ValueError, match=message):
        run_model_compile_process_task(context, {"model_name": "LapicqueNeuron"})

    assert context.artifacts == ()
