# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — COBA LIF five-backend benchmark contract tests

"""Production-path tests for the controlled COBA LIF benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from benchmarks import bench_model_coba_lif as benchmark
from sc_neurocore.accel import coba_lif as backends


def _passing_safety() -> dict[str, object]:
    return {"command": "focused", "passed": True, "returncode": 0, "output_tail": []}


def test_real_benchmark_writes_complete_conductance_parity_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all five public batch dispatchers with non-zero conductance events."""
    monkeypatch.setattr(benchmark, "N_STEPS", 5)
    monkeypatch.setattr(benchmark, "N_REPEATS", 1)
    monkeypatch.setattr(benchmark, "WARMUP_STEPS", 1)
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)
    output = tmp_path / "coba-lif.json"

    assert benchmark.main(["--json", str(output), "--allow-unpinned"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == "coba_lif_coupled_rk4_conductance_batch"
    assert payload["workload"]["n_steps"] == 5
    assert payload["workload"]["delta_ge"] > 0.0
    assert payload["workload"]["delta_gi"] > 0.0
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    assert all(row["event_count_matches_python"] for row in payload["backends"].values())
    assert all(
        row["parity_max_abs_diff"] <= benchmark.TRACE_ATOL for row in payload["backends"].values()
    )
    assert set(payload["backends"]["python"]["final_state"]) == {
        "v",
        "g_e",
        "g_i",
        "refractory_time",
    }


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
    """Pin every declared source digest to the live file bytes."""
    hashes = benchmark._source_hashes()
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected
        assert len(expected) == 64


def test_committed_evidence_matches_live_sources_and_full_parity() -> None:
    """Bind the measured five-lane artifact to current code and behavior."""
    artifact = benchmark.REPOSITORY / "benchmarks/results/local_python_2026-06-18_coba_lif_rk4.json"
    artifact_text = artifact.read_text(encoding="utf-8")
    assert str(benchmark.REPOSITORY) not in artifact_text
    payload = json.loads(artifact_text)

    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["evidence_class"] == "local_regression_single_cpu_affinity_non_exclusive"
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert payload["workload"] == {
        "current": benchmark.CURRENT,
        "delta_ge": benchmark.DELTA_GE,
        "delta_gi": benchmark.DELTA_GI,
        "initial_state": {
            "v": -59.0,
            "g_e": 1.25,
            "g_i": 0.75,
            "refractory_time": 0.3,
        },
        "n_steps": benchmark.N_STEPS,
        "parameters": "complete non-default 15-double native ABI configuration",
        "repeats": benchmark.N_REPEATS,
        "trace_atol": benchmark.TRACE_ATOL,
        "warmup_steps": benchmark.WARMUP_STEPS,
    }
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    measured_order = sorted(
        benchmark.BACKENDS,
        key=lambda backend: float(payload["backends"][backend]["median_call_ms"]),
    )
    native_order = [backend for backend in measured_order if backend != "python"]
    assert payload["measured_order"] == measured_order
    assert payload["recommended_auto_backend"] == native_order[0]
    assert payload["verification"]["rust_safety"]["passed"] is True
    assert payload["meta"]["single_cpu_pinned"] is True
    assert payload["meta"]["exclusive_cpu_isolation_claimed"] is False

    reference_events = payload["backends"]["python"]["event_count"]
    for backend in benchmark.BACKENDS:
        row = payload["backends"][backend]
        assert row["available"] is True
        assert row["used"] is True
        assert row["event_count"] == reference_events
        assert row["event_count_matches_python"] is True
        assert row["parity_max_abs_diff"] <= benchmark.TRACE_ATOL
        assert set(row["final_state"]) == {"v", "g_e", "g_i", "refractory_time"}

    hashes = payload["source_hashes"]
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected


def test_real_rust_safety_gate_executes_enrolled_module() -> None:
    """Compile and execute the benchmark's real Rust-safety module."""
    result = benchmark._verify_rust_safety()
    assert result["passed"] is True
    assert result["returncode"] == 0
    assert "coba_lif.rs" in str(result["command"])
    assert any("5 passed" in line for line in result["output_tail"])


def test_unpinned_run_is_rejected_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require explicit acknowledgement for multi-CPU affinity."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
    assert benchmark.main(["--json", str(tmp_path / "unused.json")]) == 2


def test_missing_backend_is_rejected_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse to publish a partial report without an explicit override."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(
        benchmark,
        "_probe_backend",
        lambda backend: (backend == "python", "missing" if backend != "python" else ""),
    )
    assert benchmark.main(["--json", str(tmp_path / "unused.json")]) == 2


@pytest.mark.parametrize(
    ("compiled_trace", "compiled_spikes", "expected"),
    [(np.array([0.0]), 1, 3), (np.array([1.0]), 0, 4)],
)
def test_parity_contract_failures_have_distinct_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_trace: npt.NDArray[np.float64],
    compiled_spikes: int,
    expected: int,
) -> None:
    """Distinguish event failure from bounded-state failure."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python", "mojo"))
    monkeypatch.setattr(benchmark, "N_STEPS", 1)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)

    def measured(
        backend: str,
    ) -> tuple[
        float,
        float,
        float,
        list[float],
        npt.NDArray[np.float64],
        int,
        tuple[float, float, float, float],
    ]:
        if backend == "python":
            return 1.0, 1.0, 1.0, [1.0], np.array([0.0]), 0, (0.0, 0.0, 0.0, 0.0)
        return (
            0.5,
            0.5,
            0.5,
            [0.5],
            compiled_trace,
            compiled_spikes,
            (0.0, 0.0, 0.0, 0.0),
        )

    monkeypatch.setattr(benchmark, "_measure_backend", measured)
    assert benchmark.main(["--json", str(tmp_path / f"failure-{expected}.json")]) == expected


def test_backend_probes_report_each_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind every backend name to its real availability probe."""
    monkeypatch.setattr(backends, "_HAS_RUST", True)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: True)
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)

    assert benchmark._probe_backend("python") == (True, "")
    assert benchmark._probe_backend("rust") == (True, "")
    assert benchmark._probe_backend("julia")[0] is False
    assert benchmark._probe_backend("go") == (True, "")
    assert benchmark._probe_backend("mojo")[0] is False
