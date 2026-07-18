# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire benchmark evidence gates

"""Validate runtime evidence, source binding, and fail-closed benchmark gates."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest

from benchmarks import bench_model_resonate_and_fire as benchmark


def _passing_safety() -> dict[str, object]:
    return {"command": ["focused"], "passed": True, "returncode": 0, "output_tail": []}


def _constant_result(value: float, n_steps: int) -> benchmark.BenchmarkResult:
    return {
        "x": np.full(n_steps, value, dtype=np.float64),
        "y": np.full(n_steps, value, dtype=np.float64),
        "spikes": np.zeros(n_steps, dtype=np.float64),
        "x_final": value,
        "y_final": value,
        "spike_count": 0,
    }


def test_every_public_runtime_matches_configured_python_trace() -> None:
    reference = benchmark._run_backend("python", 64)
    for backend in benchmark.BACKENDS[1:]:
        tolerance = benchmark.PARITY_ATOL[backend]
        actual = benchmark._run_backend(backend, 64)
        for key in benchmark.TRACE_KEYS:
            np.testing.assert_allclose(actual[key], reference[key], rtol=0.0, atol=tolerance)
        for key in benchmark.FINAL_KEYS:
            assert float(actual[key]) == pytest.approx(float(reference[key]), abs=tolerance)
        assert actual["spike_count"] == reference["spike_count"]


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
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
    records = benchmark._binary_hashes()
    assert set(records) == {"rust_extension", "go_shared_library", "mojo_shared_library"}
    extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    paths = {
        "rust_extension": Path(str(extension.__file__)),
        "go_shared_library": (
            benchmark.REPOSITORY
            / "src/sc_neurocore/accel/go/resonate_and_fire/libresonate_and_fire.so"
        ),
        "mojo_shared_library": (
            benchmark.REPOSITORY
            / "src/sc_neurocore/accel/mojo/resonate_and_fire/libresonate_and_fire.so"
        ),
    }
    assert str(records["rust_extension"]["path"]).startswith("$WHEEL_SITE/")
    for name, record in records.items():
        path = paths[name]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["size_bytes"] == path.stat().st_size


def test_environment_separates_loaded_runtime_and_path_cli_provenance() -> None:
    environment = benchmark._environment()
    assert environment["numpy"] == np.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(environment["julia_runtime"]))
    assert str(environment["julia_cli"]).startswith("julia version ")
    assert str(environment["mojo_pixi"]).startswith("Mojo ")
    assert str(environment["mojo_cli"]).startswith("Mojo ")
    go_binary = environment["go_binary"]
    assert isinstance(go_binary, dict)
    assert str(go_binary["go_version"]).startswith("go1.")
    assert go_binary["package"] == "github.com/anulum/sc-neurocore/accel/resonate_and_fire"
    assert go_binary["cgo_enabled"] == "1"
    assert str(environment["go_cli"]).startswith("go version ")


def test_committed_evidence_matches_sources_and_bounded_parity() -> None:
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
        assert row["spike_count_matches_python"] is True
    assert payload["source_hashes"] == benchmark._source_hashes()
    committed_binaries = payload["binary_hashes"]
    live_binaries = benchmark._binary_hashes()
    assert set(committed_binaries) == set(live_binaries)
    for name in live_binaries:
        recorded = committed_binaries[name]
        assert len(str(recorded["sha256"])) == 64
        assert isinstance(recorded["size_bytes"], int) and recorded["size_bytes"] > 0
        assert isinstance(recorded["path"], str) and recorded["path"]


def test_unpinned_run_is_rejected_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output), "--steps", "1", "--repeats", "1"]) == 2
    assert not output.exists()


def test_missing_runtime_is_rejected_by_default(
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

    def measured(
        backend: str,
        n_steps: int,
        _repeats: int,
    ) -> tuple[list[int], benchmark.BenchmarkResult]:
        return [1], _constant_result(0.8 if backend == "mojo" else 0.2, n_steps)

    monkeypatch.setattr(benchmark, "_measure_backend", measured)
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output), "--steps", "2", "--repeats", "1"]) == 3
    assert not output.exists()


def test_rust_safety_failure_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(
        benchmark,
        "_verify_rust_safety",
        lambda: {"passed": False, "returncode": 1},
    )
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output), "--steps", "2", "--repeats", "1"]) == 4
    assert not output.exists()
