# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IQIF five-backend benchmark contract tests

"""Production-path tests for the controlled Wu et al. IQIF benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import numpy.typing as npt
import pytest

from benchmarks import bench_model_iqif as benchmark
from sc_neurocore.accel import iqif as backends


def _passing_safety() -> dict[str, object]:
    """Return one successful focused Rust-safety result."""
    return {"command": "focused", "passed": True, "returncode": 0, "output_tail": []}


def test_real_benchmark_writes_five_backend_integer_parity_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all five public dispatchers with the pinned source prefix."""
    monkeypatch.setattr(benchmark, "N_STEPS", 15)
    monkeypatch.setattr(benchmark, "N_REPEATS", 1)
    monkeypatch.setattr(benchmark, "WARMUP_STEPS", 1)
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)
    output = tmp_path / "iqif.json"

    assert benchmark.main(["--json", str(output), "--allow-unpinned"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["workload"]["n_steps"] == 15
    assert payload["workload"]["current"] == 10
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    assert payload["auto_backend_order"][-1] == "python"
    reference_hash = payload["backends"]["python"]["trace_sha256"]
    assert all(row["trace_matches_python"] for row in payload["backends"].values())
    assert all(row["event_count_matches_python"] for row in payload["backends"].values())
    assert all(row["final_state_matches_python"] for row in payload["backends"].values())
    assert all(row["trace_sha256"] == reference_hash for row in payload["backends"].values())
    assert payload["backends"]["python"]["event_count"] == 1
    assert payload["backends"]["python"]["final_state"] == {"v": 128}


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
    """Pin every declared source digest to the live file bytes."""
    hashes = benchmark._source_hashes()
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected
        assert len(expected) == 64


def test_binary_hashes_bind_loaded_native_artifacts() -> None:
    """Pin every measured native lane to the exact loaded file bytes."""
    records = benchmark._binary_hashes()
    assert set(records) == {"rust_extension", "go_shared_library", "mojo_shared_library"}
    for record in records.values():
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = benchmark.REPOSITORY / path
        assert path.is_file()
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["size_bytes"] == path.stat().st_size


def test_committed_evidence_matches_live_sources_and_full_parity() -> None:
    """Bind the measured five-lane artefact to current code and behaviour."""
    artifact = benchmark.REPOSITORY / "benchmarks/results/local_python_2026-07-14_iqif.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["evidence_class"] == "local_regression_single_cpu_affinity_non_exclusive"
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert payload["workload"] == {
        "current": benchmark.CURRENT,
        "initial_state": {
            "a": 1,
            "b": 1,
            "v": 128,
            "v_max": 255,
            "v_min": 0,
            "v_reset": 128,
            "v_rest": 128,
            "v_threshold": 200,
        },
        "n_steps": benchmark.N_STEPS,
        "parameters": "pinned Wu et al. tutorial defaults; complete signed-int32 ABI",
        "repeats": benchmark.N_REPEATS,
        "warmup_steps": benchmark.WARMUP_STEPS,
    }
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    native_order = [name for name in payload["measured_order"] if name != "python"]
    assert payload["fastest_measured_native_backend"] == native_order[0]
    assert payload["auto_backend_order"] == [*native_order, "python"]
    assert payload["recommended_auto_backend"] == native_order[0]
    assert payload["auto_backend_selection_basis"] == (
        "same-host measured warm batch order; non-exclusive timings remain diagnostic"
    )
    assert payload["verification"]["rust_safety"]["passed"] is True
    assert payload["meta"]["single_cpu_pinned"] is True
    assert payload["meta"]["exclusive_cpu_isolation_claimed"] is False

    reference = payload["backends"]["python"]
    assert reference["event_count"] == 13_333
    assert reference["final_state"] == {"v": 165}
    for backend in benchmark.BACKENDS:
        row = payload["backends"][backend]
        assert row["available"] is True
        assert row["used"] is True
        assert row["trace_sha256"] == reference["trace_sha256"]
        assert row["trace_matches_python"] is True
        assert row["trace_mismatch_count"] == 0
        assert row["event_count"] == reference["event_count"]
        assert row["event_count_matches_python"] is True
        assert row["final_state"] == reference["final_state"]
        assert row["final_state_matches_python"] is True
        assert row["parity_max_abs_diff"] == 0

    hashes = payload["source_hashes"]
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected

    # Native artifacts are rebuilt per environment; their exact bytes are a
    # reproducible-build property, not part of this local-regression fidelity
    # claim (evidence_class is local_regression, production_speed_claim is
    # False). Bind the committed provenance shape and confirm the current tree
    # still produces the three loadable native lanes, rather than demanding
    # cross-build byte-for-byte reproducibility (see
    # test_binary_hashes_bind_loaded_native_artifacts for live self-consistency).
    committed_binaries = payload["binary_hashes"]
    live_binaries = benchmark._binary_hashes()
    assert set(committed_binaries) == set(live_binaries)
    assert set(live_binaries) == {"rust_extension", "go_shared_library", "mojo_shared_library"}
    for name in live_binaries:
        recorded = committed_binaries[name]
        assert len(str(recorded["sha256"])) == 64
        assert isinstance(recorded["size_bytes"], int)
        assert recorded["size_bytes"] > 0
        assert isinstance(recorded["path"], str)
        assert recorded["path"]


def test_real_rust_safety_gate_executes_enrolled_module() -> None:
    """Compile and execute the benchmark's real Rust-safety module."""
    result = benchmark._verify_rust_safety()
    assert result["passed"] is True
    assert result["returncode"] == 0
    assert "iqif.rs" in str(result["command"])
    assert any("8 passed" in line for line in result["output_tail"])


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
    ("compiled_trace", "compiled_spikes", "compiled_final"),
    [
        (np.array([129], dtype=np.int64), 0, 128),
        (np.array([128], dtype=np.int64), 1, 128),
        (np.array([128], dtype=np.int64), 0, 129),
    ],
)
def test_trace_count_and_final_state_parity_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_trace: npt.NDArray[np.int64],
    compiled_spikes: int,
    compiled_final: int,
) -> None:
    """Reject trajectory, event-count, and final-state divergence."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python", "mojo"))
    monkeypatch.setattr(benchmark, "N_STEPS", 1)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)

    def measured(
        backend: str,
    ) -> tuple[float, float, float, list[float], npt.NDArray[np.int64], int, int]:
        if backend == "python":
            return 1.0, 1.0, 1.0, [1.0], np.array([128], dtype=np.int64), 0, 128
        return 0.5, 0.5, 0.5, [0.5], compiled_trace, compiled_spikes, compiled_final

    monkeypatch.setattr(benchmark, "_measure_backend", measured)
    output = tmp_path / f"failure-{compiled_spikes}-{compiled_final}.json"
    assert benchmark.main(["--json", str(output)]) == 3


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


def test_host_metadata_helpers_use_portable_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Linux metadata and PATH entries remain explicit in evidence."""
    original_read_text = Path.read_text

    def selective_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path == Path("/proc/cpuinfo") or path == tmp_path / "missing":
            raise OSError("metadata unavailable")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", selective_read)
    monkeypatch.setattr(platform, "processor", lambda: "")
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    fallback = tmp_path / "runtime"
    fallback.touch()
    assert benchmark._cpu_model() == "unknown"
    assert benchmark._read_optional(tmp_path / "missing") == "unavailable"
    assert benchmark._tool_path("runtime", fallback) == str(fallback)
    assert benchmark._tool_path("runtime", tmp_path / "absent") is None


def test_tool_version_reports_empty_and_failed_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime provenance distinguishes absence and execution failure."""
    assert benchmark._tool_version([]) == "unavailable"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot execute")),
    )
    assert benchmark._tool_version(["missing"]) == "unavailable"

    completed = subprocess.CompletedProcess(["runtime"], 7, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: completed)
    assert benchmark._tool_version(["runtime"]) == "exit 7"


def test_rust_safety_gate_reports_compile_and_execution_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmark cannot promote when its standalone safety proof fails."""
    failure = subprocess.CompletedProcess(["rustc"], 1, stdout="", stderr="compile failed")
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: failure)
    result = benchmark._verify_rust_safety()
    assert result["passed"] is False
    assert result["returncode"] == 1
    assert result["output_tail"] == ["compile failed"]

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("rustc unavailable")),
    )
    result = benchmark._verify_rust_safety()
    assert result["passed"] is False
    assert result["returncode"] == -1
    assert result["output_tail"] == ["rustc unavailable"]


def _measured_python(
    _backend: str,
) -> tuple[float, float, float, list[float], npt.NDArray[np.int64], int, int]:
    """Return one deterministic benchmark row for CLI gate tests."""
    return 1.0, 1.0, 1.0, [1.0], np.array([128], dtype=np.int64), 0, 128


def test_allowed_unavailable_backend_is_recorded_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit diagnostic override retains an unavailable backend row."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python", "mojo"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(
        benchmark,
        "_probe_backend",
        lambda backend: (backend == "python", "missing" if backend == "mojo" else ""),
    )
    monkeypatch.setattr(benchmark, "_measure_backend", _measured_python)
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)
    output = tmp_path / "partial.json"

    assert benchmark.main(["--json", str(output), "--allow-unavailable-backends"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backends"]["mojo"] == {
        "available": False,
        "used": False,
        "unavailable_reason": "missing",
    }


def test_python_reference_must_be_measured_before_native_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a benchmark configuration that measures a native lane first."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("mojo", "python"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_measure_backend", _measured_python)

    with pytest.raises(RuntimeError, match="Python reference must be measured first"):
        benchmark.main(["--json", str(tmp_path / "invalid.json")])


def test_failed_rust_safety_gate_returns_dedicated_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed safety module blocks an otherwise matching report."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python",))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_measure_backend", _measured_python)
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(
        benchmark,
        "_verify_rust_safety",
        lambda: {"command": "focused", "passed": False, "returncode": 1, "output_tail": []},
    )

    assert benchmark.main(["--json", str(tmp_path / "failed.json")]) == 5
