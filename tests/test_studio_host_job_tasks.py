# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — hosted worker outcome and artifact tests

"""Exercise public worker adapters with real contexts and the native compiler."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from sc_neurocore.studio.api.host_job_tasks import run_host_analysis_task, run_host_compiler_task
from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.jobs_models import StudioJobCancelled, StudioJobRefused


@pytest.fixture
def context(tmp_path: Path) -> StudioJobContext:
    """Supply actual confined cancellation and artifact custody."""
    return StudioJobContext(
        job_id="sj_1234567890abcdef",
        work_dir=tmp_path,
        cancel_event=threading.Event(),
        max_artifact_bytes=1 << 20,
    )


def build_request() -> dict[str, object]:
    """Provide ordered authored source with exact maximum key and stream."""
    return {
        "schema_version": "studio.equation-ir-request.v1",
        "model": {
            "equations": [{"name": "z", "expression": "I-a"}, {"name": "a", "expression": "z-a"}],
            "initial_state": [{"name": "z", "value": 3}, {"name": "a", "value": 1}],
            "method": "gauss_seidel",
            "rng_seed": "18446744073709551615",
            "random_stream": 4294967295,
        },
        "profile": {
            "module_name": "host_worker_source",
            "data_width": 32,
            "fraction": 16,
            "function_lowering": "v2",
        },
    }


def test_native_build_verify_replay_emission_keep_actual_outcomes(
    context: StudioJobContext,
) -> None:
    """Four real native operations publish matching digests without invented acceptance."""
    built = run_host_compiler_task(
        context,
        {"envelope_json": json.dumps({"operation": "ir.build", "request": build_request()})},
    )
    assert built["accepted"] is True
    graph = built["result"]
    assert isinstance(graph, dict)
    assert graph["graph_name"] == "host_worker_source"
    assert graph["source_present"] is True
    for operation in ["ir.verify", "ir.replay", "ir.emit-sv"]:
        outcome = run_host_compiler_task(
            context,
            {
                "envelope_json": json.dumps(
                    {"operation": operation, "request": {"ir_text": graph["ir_text"]}}
                )
            },
        )
        assert outcome["accepted"] is True
        result = outcome["result"]
        assert isinstance(result, dict)
        assert result["ir_text"] == graph["ir_text"]
        if operation != "ir.verify":
            assert result["source_replay"] == "passed"
        if operation == "ir.emit-sv":
            assert isinstance(result["systemverilog"], str)
            assert "module host_worker_source" in result["systemverilog"]
        artifact = context.artifacts[-1]
        assert artifact.relative_path == "host/result.json"
        encoded = json.dumps(outcome, allow_nan=False).encode()
        assert artifact.size_bytes == len(encoded)
        assert artifact.sha256 == hashlib.sha256(encoded).hexdigest()


def test_original_partial_power_source_reaches_hosted_worker_and_native_replay(
    context: StudioJobContext,
) -> None:
    """The hosted boundary retains the exact original partial-state compiler counterexample."""
    from sc_neurocore_engine.ir import parse_ir

    request = {
        "schema_version": "studio.equation-ir-request.v1",
        "model": {
            "equations": [{"name": "v", "expression": "I**1.5+clip(I,.25,1)**clip(w,-1,1)"}],
            "initial_state": [{"name": "w", "value": 0.5}],
            "method": "map",
            "dt": {"value": 1},
            "rng_seed": "18446744073709551615",
            "random_stream": 4294967295,
        },
        "profile": {"function_lowering": "v2"},
    }
    result = run_host_compiler_task(
        context,
        {"envelope_json": json.dumps({"operation": "ir.build", "request": request})},
    )
    assert result["accepted"] is True
    graph = result["result"]
    assert isinstance(graph, dict)
    text = graph["ir_text"]
    assert isinstance(text, str)
    source = json.loads(str(parse_ir(text).source_payload))
    assert source["model"]["initial_state"] == [["w", 0.5], ["v", 0.0]]
    assert source["model"]["equations"] == [["v", "I**1.5+clip(I,.25,1)**clip(w,-1,1)"]]
    assert source["model"]["rng_seed"] == 2**64 - 1
    assert source["model"]["random_stream"] == 2**32 - 1
    for operation in ("ir.replay", "ir.emit-sv"):
        replay = run_host_compiler_task(
            context,
            {"envelope_json": json.dumps({"operation": operation, "request": {"ir_text": text}})},
        )
        assert replay["accepted"] is True
        outcome = replay["result"]
        assert isinstance(outcome, dict)
        assert outcome["source_replay"] == "passed"
        assert outcome["ir_text"] == text
        if operation == "ir.emit-sv":
            rtl = outcome["systemverilog"]
            assert isinstance(rtl, str)
            assert "w_out" in rtl


def test_native_refusal_is_preserved_in_a_completed_worker_artifact(
    context: StudioJobContext,
) -> None:
    """Invalid native graph cannot be labelled a successful compilation."""
    result = run_host_compiler_task(
        context,
        {
            "envelope_json": json.dumps(
                {"operation": "ir.verify", "request": {"ir_text": "invalid native graph"}}
            )
        },
    )
    assert result["accepted"] is False
    assert result["result"] is None
    refusal = result["refusal"]
    assert isinstance(refusal, dict)
    assert refusal["code"] == "invalid_input"
    assert [artifact.relative_path for artifact in context.artifacts] == [
        "equation-ir/result.json",
        "equation-ir/evidence.json",
        "host/result.json",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"envelope_json": []},
        {"envelope_json": "{}"},
        {"envelope_json": "{}", "owner": "caller"},
    ],
)
def test_invalid_worker_envelope_never_writes_a_result(
    context: StudioJobContext, payload: Mapping[str, object]
) -> None:
    """Process inputs must match the complete closed admission contract again."""
    with pytest.raises(ValueError):
        run_host_analysis_task(context, payload)
    assert context.artifacts == ()


def test_worker_families_refuse_crossed_admissions(context: StudioJobContext) -> None:
    """Static task identity cannot silently run a different scientific family."""
    with pytest.raises(ValueError, match="analysis operation"):
        run_host_analysis_task(
            context,
            {"envelope_json": json.dumps({"operation": "ir.build", "request": build_request()})},
        )
    with pytest.raises(ValueError, match="compiler operation"):
        run_host_compiler_task(
            context,
            {
                "envelope_json": json.dumps(
                    {"operation": "simulate", "request": {"equations": ["dv/dt=-v"], "duration": 1}}
                )
            },
        )
    assert context.artifacts == ()


def test_real_analysis_keeps_order_and_refuses_small_artifact_budget(tmp_path: Path) -> None:
    """A real sensitivity result retains source order; oversized publication refuses."""
    payload = {
        "envelope_json": json.dumps(
            {
                "operation": "sensitivity",
                "request": {
                    "equations": ["dv/dt=I-z*v+a"],
                    "params": {"z": 1, "a": 2},
                    "init": {"v": 0},
                    "duration": 1,
                    "dt": 0.1,
                },
            }
        )
    }
    context = StudioJobContext(
        job_id="sj_1234567890abcdef",
        work_dir=tmp_path / "normal",
        cancel_event=threading.Event(),
        max_artifact_bytes=1 << 20,
    )
    outcome = run_host_analysis_task(context, payload)
    rows = outcome["sensitivities"]
    assert isinstance(rows, list)
    assert [row["param"] for row in rows] == ["z", "a"]
    assert context.artifacts[-1].relative_path == "host/result.json"
    limited = StudioJobContext(
        job_id="sj_1234567890abcdef",
        work_dir=tmp_path / "limited",
        cancel_event=threading.Event(),
        max_artifact_bytes=1,
    )
    with pytest.raises(StudioJobRefused, match="size limit"):
        run_host_analysis_task(limited, payload)
    assert limited.artifacts == ()


def test_cancelled_workers_do_not_execute_or_publish(tmp_path: Path) -> None:
    """Both entry points honor an actual cancellation event before execution."""
    event = threading.Event()
    event.set()
    context = StudioJobContext(
        job_id="sj_1234567890abcdef",
        work_dir=tmp_path,
        cancel_event=event,
        max_artifact_bytes=1 << 20,
    )
    for task in [run_host_analysis_task, run_host_compiler_task]:
        with pytest.raises(StudioJobCancelled):
            task(context, {})
    assert context.artifacts == ()
