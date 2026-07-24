# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (gates_and_rejects) from former test_bench_iqif.py

from __future__ import annotations

from tests.bench_iqif_support import *  # noqa: F403


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
