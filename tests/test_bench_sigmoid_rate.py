# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sigmoid-rate benchmark evidence gates

"""Validate five-backend evidence, source binding, and fail-closed CLI gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from benchmarks import bench_model_sigmoid_rate as benchmark


def _passing_safety() -> dict[str, object]:
    return {"command": ["focused"], "passed": True, "returncode": 0, "output_tail": []}


def test_every_public_backend_matches_configured_python_trace() -> None:
    """Execute a bounded real batch through all five public dispatchers."""
    reference, reference_final = benchmark._run_backend("python", 64)
    for backend in benchmark.BACKENDS[1:]:
        actual, final_rate = benchmark._run_backend(backend, 64)
        np.testing.assert_allclose(actual, reference, rtol=0.0, atol=benchmark.PARITY_ATOL)
        assert final_rate == pytest.approx(reference_final, rel=0.0, abs=benchmark.PARITY_ATOL)


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
    """Pin every declared source digest to the live file bytes."""
    hashes = benchmark._source_hashes()
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected
        stem, suffix = relative.rsplit(".", 1)
        assert hashes[stem] == {suffix: expected}


def test_binary_hashes_bind_loaded_native_artifacts() -> None:
    """Pin every measured native object to exact local bytes."""
    records = benchmark._binary_hashes()
    assert set(records) == {"rust_extension", "go_shared_library", "mojo_shared_library"}
    for record in records.values():
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = benchmark.REPOSITORY / path
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["size_bytes"] == path.stat().st_size


def test_committed_evidence_matches_sources_and_full_parity() -> None:
    """Bind the published five-lane artifact to current code and behaviour."""
    artifact = benchmark.DEFAULT_OUTPUT
    artifact_text = artifact.read_text(encoding="utf-8")
    assert str(benchmark.REPOSITORY) not in artifact_text
    payload = json.loads(artifact_text)
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
        assert row["available"] is True
        assert row["used"] is True
        assert row["trace_matches_python"] is True
        assert row["trace_mismatch_count"] == 0
        assert row["parity_max_abs_diff"] <= benchmark.PARITY_ATOL
        assert row["final_rate_matches_python"] is True
    assert payload["source_hashes"] == benchmark._source_hashes()
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


def test_unpinned_run_is_rejected_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require explicit acknowledgement for multi-CPU affinity."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output), "--steps", "1", "--repeats", "1"]) == 2
    assert not output.exists()


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
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output), "--steps", "1", "--repeats", "1"]) == 2
    assert not output.exists()


def test_trace_or_final_divergence_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a native trajectory outside the declared float tolerance."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python", "mojo"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_binary_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_environment", lambda: {})
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)

    def measured(
        backend: str,
        _steps: int,
        _repeats: int,
    ) -> tuple[list[int], npt.NDArray[np.float64], float]:
        if backend == "python":
            return [10], np.asarray([0.25]), 0.25
        return [5], np.asarray([0.5]), 0.5

    monkeypatch.setattr(benchmark, "_measure_backend", measured)
    output = tmp_path / "failure.json"
    assert benchmark.main(["--json", str(output), "--steps", "1", "--repeats", "1"]) == 3
    assert not output.exists()


def test_real_rust_safety_gate_executes_enrolled_module() -> None:
    """Compile and run the benchmark's real standalone safety tests."""
    result = benchmark._verify_rust_safety()
    assert result["passed"] is True
    assert result["returncode"] == 0
    output_tail = result["output_tail"]
    assert isinstance(output_tail, list)
    assert any(isinstance(line, str) and "8 passed" in line for line in output_tail)
