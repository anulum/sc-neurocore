# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — EscapeRate five-backend benchmark contract tests

"""Production-path tests for the controlled EscapeRate benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from benchmarks import bench_model_escape_rate as benchmark
from sc_neurocore.accel import escape_rate as backends


def _passing_safety() -> dict[str, object]:
    return {"command": "focused", "passed": True, "returncode": 0, "output_tail": []}


def test_real_benchmark_writes_seeded_five_backend_parity_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all five public batch dispatchers with one exact seeded trace."""
    monkeypatch.setattr(benchmark, "N_STEPS", 5)
    monkeypatch.setattr(benchmark, "N_REPEATS", 1)
    monkeypatch.setattr(benchmark, "WARMUP_STEPS", 1)
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)
    output = tmp_path / "escape-rate.json"

    assert benchmark.main(["--json", str(output), "--allow-unpinned"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["workload"]["n_steps"] == 5
    assert payload["workload"]["initial_state"] == {"v": -64.0, "rng_state": 0x1234}
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    assert all(row["event_count_matches_python"] for row in payload["backends"].values())
    assert all(row["rng_state_matches_python"] for row in payload["backends"].values())
    assert all(
        row["parity_max_abs_diff"] <= benchmark.TRACE_ATOL for row in payload["backends"].values()
    )
    assert set(payload["backends"]["python"]["final_state"]) == {"v", "rng_state"}


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
    """Pin every declared source digest to the live file bytes."""
    hashes = benchmark._source_hashes()
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected
        assert len(expected) == 64


def test_committed_evidence_matches_live_sources_and_full_parity() -> None:
    """Bind the measured five-lane artifact to current code and behavior."""
    artifact = (
        benchmark.REPOSITORY / "benchmarks/results/local_python_2026-07-14_escape_rate_lfsr16.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["evidence_class"] == "local_regression_single_cpu_affinity_non_exclusive"
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert payload["workload"] == {
        "current": benchmark.CURRENT,
        "initial_state": {"v": -64.0, "rng_state": 0x1234},
        "n_steps": benchmark.N_STEPS,
        "parameters": "complete non-default 9-double plus seeded LFSR16 native ABI",
        "repeats": benchmark.N_REPEATS,
        "trace_atol": benchmark.TRACE_ATOL,
        "warmup_steps": benchmark.WARMUP_STEPS,
    }
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    assert payload["recommended_auto_backend"] in benchmark.BACKENDS[1:]
    assert payload["recommended_auto_backend"] == next(
        backend for backend in payload["measured_order"] if backend != "python"
    )
    assert payload["verification"]["rust_safety"]["passed"] is True
    assert payload["meta"]["single_cpu_pinned"] is True
    assert payload["meta"]["exclusive_cpu_isolation_claimed"] is False

    reference_events = payload["backends"]["python"]["event_count"]
    reference_rng = payload["backends"]["python"]["final_state"]["rng_state"]
    for backend in benchmark.BACKENDS:
        row = payload["backends"][backend]
        assert row["available"] is True
        assert row["used"] is True
        assert row["event_count"] == reference_events
        assert row["event_count_matches_python"] is True
        assert row["final_state"]["rng_state"] == reference_rng
        assert row["rng_state_matches_python"] is True
        assert row["parity_max_abs_diff"] <= benchmark.TRACE_ATOL
        assert set(row["final_state"]) == {"v", "rng_state"}

    hashes = payload["source_hashes"]
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected


def test_real_rust_safety_gate_executes_enrolled_module() -> None:
    """Compile and execute the benchmark's real Rust-safety module."""
    result = benchmark._verify_rust_safety()
    assert result["passed"] is True
    assert result["returncode"] == 0
    assert "escape_rate.rs" in str(result["command"])
    assert any("7 passed" in line for line in result["output_tail"])


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
    ("compiled_trace", "compiled_spikes", "compiled_state", "expected"),
    [
        (np.array([0.0]), 1, (0.0, 1), 3),
        (np.array([0.0]), 0, (0.0, 2), 3),
        (np.array([1.0]), 0, (0.0, 1), 4),
    ],
)
def test_parity_contract_failures_have_distinct_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_trace: npt.NDArray[np.float64],
    compiled_spikes: int,
    compiled_state: tuple[float, int],
    expected: int,
) -> None:
    """Distinguish exact event/RNG failure from bounded voltage failure."""
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
        tuple[float, int],
    ]:
        if backend == "python":
            return 1.0, 1.0, 1.0, [1.0], np.array([0.0]), 0, (0.0, 1)
        return 0.5, 0.5, 0.5, [0.5], compiled_trace, compiled_spikes, compiled_state

    monkeypatch.setattr(benchmark, "_measure_backend", measured)
    output = tmp_path / f"failure-{expected}-{compiled_spikes}-{compiled_state[1]}.json"
    assert benchmark.main(["--json", str(output)]) == expected


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
