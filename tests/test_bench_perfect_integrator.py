# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect Integrator five-backend benchmark contract tests

"""Production-path tests for the controlled Perfect Integrator benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from benchmarks import bench_model_perfect_integrator as benchmark
from sc_neurocore.accel import perfect_integrator as backends


def test_real_benchmark_writes_complete_parity_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every measured backend through the real simulation API."""
    monkeypatch.setattr(benchmark, "N_STEPS", 3)
    monkeypatch.setattr(benchmark, "N_REPEATS", 1)
    output = tmp_path / "perfect-integrator.json"

    assert benchmark.main(["--json", str(output), "--allow-unpinned"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == "perfect_integrator_naud_gerstner_2012_exact_complete_packet"
    assert payload["workload"]["n_steps"] == 3
    assert set(payload["measured_order"]) == {"python", "rust", "julia", "go", "mojo"}
    assert all(row["event_vector_matches_python"] for row in payload["backends"].values())
    assert set(payload["backends"]["python"]["final_state"]) == {"v"}


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
    """Pin every declared source digest to the live file bytes."""
    hashes = benchmark._source_hashes()
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected
        assert len(expected) == 64


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
    ("compiled_trace", "compiled_events", "expected"),
    [
        (np.array([0.0]), np.array([1], dtype=np.uint8), 3),
        (np.array([1.0]), np.array([0], dtype=np.uint8), 4),
    ],
)
def test_parity_contract_failures_have_distinct_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_trace: npt.NDArray[np.float64],
    compiled_events: npt.NDArray[np.uint8],
    expected: int,
) -> None:
    """Distinguish event failure from bit-exact state failure."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python", "mojo"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})

    def measured(
        backend: str,
    ) -> tuple[
        float,
        float,
        npt.NDArray[np.float64],
        npt.NDArray[np.uint8],
        tuple[float],
    ]:
        if backend == "python":
            return 1.0, 1.0, np.array([0.0]), np.array([0], dtype=np.uint8), (0.0,)
        return 0.5, 0.5, compiled_trace, compiled_events, (0.0,)

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
