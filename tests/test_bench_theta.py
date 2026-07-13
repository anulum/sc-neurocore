# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta five-backend benchmark contract tests

"""Production-path tests for the controlled Theta benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess

import numpy as np
import numpy.typing as npt
import pytest

from benchmarks import bench_model_theta as benchmark
from sc_neurocore.accel import theta as backends


def test_real_benchmark_writes_complete_parity_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every measured backend through the real simulation API."""
    monkeypatch.setattr(benchmark, "N_STEPS", 3)
    monkeypatch.setattr(benchmark, "N_REPEATS", 1)
    monkeypatch.setattr(
        benchmark,
        "_verify_rust_safety",
        lambda: {"command": "focused", "passed": True, "returncode": 0, "output_tail": []},
    )
    output = tmp_path / "theta.json"

    assert benchmark.main(["--json", str(output), "--allow-unpinned"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["workload"]["n_steps"] == 3
    assert set(payload["measured_order"]) == {"python", "rust", "julia", "go", "mojo"}
    assert payload["dispatcher_order"] == list(benchmark.DISPATCH_ORDER)
    assert all(row["event_count_matches_python"] for row in payload["backends"].values())
    assert payload["verification"]["rust_safety"]["passed"] is True
    assert set(payload["backends"]["python"]["final_state"]) == {"theta"}


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
    """Pin every declared source digest to the live file bytes."""
    hashes = benchmark._source_hashes()
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected
        assert len(expected) == 64


def test_committed_evidence_matches_live_sources_and_bounded_parity() -> None:
    """Bind the measured five-lane artefact to current code and Python behaviour."""
    artifact = (
        benchmark.REPOSITORY / "benchmarks/results/local_python_2026-06-16_theta_exact_flow.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["workload"] == {
        "circular_phase_atol": benchmark.PHASE_ATOL,
        "current": benchmark.CURRENT,
        "n_steps": benchmark.N_STEPS,
        "parameters": "Theta factory defaults; tangent-half-angle exact constant-current flow",
        "repeats": benchmark.N_REPEATS,
    }
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    assert payload["dispatcher_order"] == list(benchmark.DISPATCH_ORDER)
    assert payload["verification"]["rust_safety"]["passed"] is True

    reference_events = payload["backends"]["python"]["event_count"]
    for backend in benchmark.BACKENDS:
        row = payload["backends"][backend]
        assert row["available"] is True
        assert row["used"] is True
        assert row["event_count"] == reference_events
        assert row["event_count_matches_python"] is True
        assert row["parity_max_circular_phase_diff"] <= benchmark.PHASE_ATOL

    hashes = payload["source_hashes"]
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected


def test_real_rust_safety_gate_executes_enrolled_module() -> None:
    """Execute the benchmark's real Rust-safety verification command."""
    result = benchmark._verify_rust_safety()
    assert result["passed"] is True
    assert result["returncode"] == 0
    assert "theta::tests" in result["command"]
    assert any("10 passed" in line for line in result["output_tail"])


def test_circular_phase_metric_handles_wrap_boundary() -> None:
    """Measure shortest-arc error rather than a false two-pi discontinuity."""
    actual = np.array([-math.pi + 1.0e-9])
    expected = np.array([math.pi - 1.0e-9])
    assert benchmark._max_circular_phase_diff(actual, expected) == pytest.approx(2.0e-9)
    assert benchmark._max_circular_phase_diff(np.array([]), np.array([])) == 0.0


def test_host_metadata_fallbacks_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report unavailable host files and a portable CPU-name fallback."""
    missing = tmp_path / "missing"
    assert benchmark._read_optional(missing) == "unavailable"

    def unreadable(_path: Path, *, encoding: str) -> str:
        assert encoding == "utf-8"
        raise OSError("host metadata unavailable")

    monkeypatch.setattr(Path, "read_text", unreadable)
    monkeypatch.setattr(platform, "processor", lambda: "portable-cpu")
    assert benchmark._cpu_model() == "portable-cpu"


def test_runtime_tool_fallbacks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve an explicit executable fallback and contain probe failures."""
    fallback = tmp_path / "runtime"
    fallback.touch()
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert benchmark._tool_path("runtime", fallback) == str(fallback)
    assert benchmark._tool_path("runtime") is None
    assert benchmark._tool_version([]) == "unavailable"

    def unavailable_run(
        _command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        raise OSError("runtime unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable_run)
    assert benchmark._tool_version([str(fallback), "--version"]) == "unavailable"


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


def test_unavailable_backend_override_records_explicit_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep acknowledged missing runtimes visible rather than silently dropping them."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python", "mojo"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(
        benchmark,
        "_probe_backend",
        lambda backend: (backend == "python", "Mojo runtime unavailable"),
    )
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})
    monkeypatch.setattr(
        benchmark,
        "_verify_rust_safety",
        lambda: {"command": "focused", "passed": True, "returncode": 0, "output_tail": []},
    )
    monkeypatch.setattr(
        benchmark,
        "_measure_backend",
        lambda _backend: (1.0, 1.0, np.array([0.0]), 0, (0.0,)),
    )
    output = tmp_path / "acknowledged-missing.json"

    assert benchmark.main(["--json", str(output), "--allow-unavailable-backends"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["measured_order"] == ["python"]
    assert payload["backends"]["mojo"] == {
        "available": False,
        "unavailable_reason": "Mojo runtime unavailable",
        "used": False,
    }


def test_compiled_lane_cannot_precede_python_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed if benchmark lane ordering no longer establishes Python first."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("mojo", "python"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(
        benchmark,
        "_measure_backend",
        lambda _backend: (1.0, 1.0, np.array([0.0]), 0, (0.0,)),
    )

    with pytest.raises(RuntimeError, match="Python reference must be measured first"):
        benchmark.main(["--json", str(tmp_path / "invalid-order.json")])


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
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})
    monkeypatch.setattr(
        benchmark,
        "_verify_rust_safety",
        lambda: {"command": "focused", "passed": True, "returncode": 0, "output_tail": []},
    )

    def measured(
        backend: str,
    ) -> tuple[float, float, npt.NDArray[np.float64], int, tuple[float]]:
        if backend == "python":
            return 1.0, 1.0, np.array([0.0]), 0, (0.0,)
        return 0.5, 0.5, compiled_trace, compiled_spikes, (0.0,)

    monkeypatch.setattr(benchmark, "_measure_backend", measured)
    assert benchmark.main(["--json", str(tmp_path / f"failure-{expected}.json")]) == expected


def test_rust_safety_failure_has_distinct_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse evidence when the actual Rust-safety module fails."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python",))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})
    monkeypatch.setattr(
        benchmark,
        "_measure_backend",
        lambda _backend: (1.0, 1.0, np.array([0.0]), 0, (0.0,)),
    )
    monkeypatch.setattr(
        benchmark,
        "_verify_rust_safety",
        lambda: {"command": "focused", "passed": False, "returncode": 1, "output_tail": []},
    )
    assert benchmark.main(["--json", str(tmp_path / "rust-failure.json")]) == 5


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
