# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wong-Wang benchmark evidence gates

"""Validate runtime evidence, source binding, and fail-closed benchmark gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks import bench_wong_wang as benchmark


def _passing_safety() -> dict[str, object]:
    """Return a focused passing safety record for CLI gate isolation."""
    return {"command": ["focused"], "passed": True, "returncode": 0, "output_tail": []}


def _constant_result(value: float, n_steps: int) -> benchmark.WongWangResult:
    """Return one complete synthetic batch result for gate tests."""
    result: benchmark.WongWangResult = {
        key: np.full(n_steps, value, dtype=np.float64) for key in benchmark.TRACE_KEYS
    }
    result.update({key: value for key in benchmark.FINAL_KEYS})
    return result


def test_every_public_runtime_matches_configured_python_trace() -> None:
    """Execute complete traces and final states on every maintained runtime."""
    reference = benchmark._run_backend("python", 64)
    for backend in benchmark.BACKENDS[1:]:
        tolerance = benchmark.PARITY_ATOL[backend]
        actual = benchmark._run_backend(backend, 64)
        for key in benchmark.TRACE_KEYS:
            np.testing.assert_allclose(actual[key], reference[key], rtol=0.0, atol=tolerance)
        for key in benchmark.FINAL_KEYS:
            assert float(actual[key]) == pytest.approx(float(reference[key]), abs=tolerance)


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
    """Bind every declared source path and its suffix-addressable alias."""
    hashes = benchmark._source_hashes()
    expected_aliases: dict[str, dict[str, str]] = {}
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected
        stem, suffix = relative.rsplit(".", 1)
        expected_aliases.setdefault(stem, {})[suffix] = expected
    for stem, aliases in expected_aliases.items():
        assert hashes[stem] == aliases


def test_binary_hashes_bind_loaded_native_artefacts() -> None:
    """Bind the exact loaded Rust extension and Go/Mojo shared libraries."""
    records = benchmark._binary_hashes()
    assert set(records) == {"rust_extension", "go_shared_library", "mojo_shared_library"}
    for record in records.values():
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = benchmark.REPOSITORY / path
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["size_bytes"] == path.stat().st_size


def test_committed_evidence_matches_sources_and_bounded_parity() -> None:
    """Reject stale evidence, unavailable runtimes, and unbounded trace drift."""
    payload = json.loads(benchmark.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert payload["meta"]["single_cpu_pinned"] is True
    assert payload["meta"]["exclusive_cpu_isolation_claimed"] is False
    assert payload["verification"]["rust_safety"]["passed"] is True
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    for backend in benchmark.BACKENDS:
        row = payload["backends"][backend]
        assert row["available"] is True and row["used"] is True
        assert row["trace_matches_python"] is True
        assert row["trace_mismatch_count"] == 0
        assert row["parity_max_abs_diff"] <= benchmark.PARITY_ATOL[backend]
        assert row["final_state_matches_python"] is True
    assert payload["source_hashes"] == benchmark._source_hashes()
    assert payload["binary_hashes"] == benchmark._binary_hashes()


def test_unpinned_run_is_rejected_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require one logical-CPU affinity unless explicitly relaxed."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output), "--steps", "1", "--repeats", "1"]) == 2
    assert not output.exists()


def test_missing_runtime_is_rejected_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not publish a partial runtime table as complete evidence."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(
        benchmark,
        "_probe_backend",
        lambda backend: (backend != "mojo", "missing") if backend == "mojo" else (True, ""),
    )
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output), "--steps", "1", "--repeats", "1"]) == 2
    assert not output.exists()


def test_parity_failure_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a failure and leave no artefact when one runtime exceeds bounds."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)
    monkeypatch.setattr(benchmark, "_binary_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_environment", lambda: {})

    def measured(
        backend: str,
        n_steps: int,
        _repeats: int,
    ) -> tuple[list[int], benchmark.WongWangResult]:
        value = 0.8 if backend == "mojo" else 0.2
        return [1], _constant_result(value, n_steps)

    monkeypatch.setattr(benchmark, "_measure_backend", measured)
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output), "--steps", "2", "--repeats", "1"]) == 3
    assert not output.exists()
