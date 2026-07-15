# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Threshold-linear benchmark evidence gates

"""Validate five-backend evidence, source binding, and fail-closed CLI gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks import bench_model_threshold_linear_rate as benchmark


def _passing_safety() -> dict[str, object]:
    return {"command": ["focused"], "passed": True, "returncode": 0, "output_tail": []}


def test_every_public_backend_matches_configured_python_trace() -> None:
    reference, reference_final = benchmark._run_backend("python", 64)
    for backend in benchmark.BACKENDS[1:]:
        actual, final_rate = benchmark._run_backend(backend, 64)
        np.testing.assert_array_equal(actual, reference)
        assert final_rate == reference_final


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
    hashes = benchmark._source_hashes()
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected
        stem, suffix = relative.rsplit(".", 1)
        assert hashes[stem] == {suffix: expected}


def test_binary_hashes_bind_loaded_native_artifacts() -> None:
    records = benchmark._binary_hashes()
    assert set(records) == {"rust_extension", "go_shared_library", "mojo_shared_library"}
    for record in records.values():
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = benchmark.REPOSITORY / path
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["size_bytes"] == path.stat().st_size


def test_committed_evidence_matches_sources_and_exact_parity() -> None:
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
        assert row["parity_max_abs_diff"] == 0.0
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
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output), "--steps", "1", "--repeats", "1"]) == 2
    assert not output.exists()


def test_missing_backend_is_rejected_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)
    monkeypatch.setattr(benchmark, "_binary_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_environment", lambda: {})

    def measured(backend: str, n_steps: int, _repeats: int):
        value = 4.0 if backend == "mojo" else 3.0
        return [1], np.full(n_steps, value), value

    monkeypatch.setattr(benchmark, "_measure_backend", measured)
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output), "--steps", "2", "--repeats", "1"]) == 3
    assert not output.exists()
