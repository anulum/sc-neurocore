# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Connor-Stevens Mojo benchmark contract tests

"""Production-path tests for the Connor-Stevens Mojo benchmark harness."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from benchmarks import bench_connor_stevens_mojo as benchmark
from sc_neurocore.neurons.models import connor_stevens

_BACKENDS_AVAILABLE = connor_stevens._HAS_RUST and connor_stevens._ensure_mojo_loaded()


@pytest.mark.skipif(not _BACKENDS_AVAILABLE, reason="Connor-Stevens Rust/Mojo backends unavailable")
def test_real_benchmark_writes_pinned_parity_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every measured backend through the real simulation API."""
    monkeypatch.setattr(benchmark, "N_STEPS", 3)
    monkeypatch.setattr(benchmark, "N_REPEATS", 1)
    output = tmp_path / "connor.json"

    assert benchmark.main(["--json", str(output), "--allow-unpinned"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == "connor_stevens_mojo_simulate"
    assert payload["workload"]["n_steps"] == 3
    assert payload["backends"]["mojo"]["event_count_matches_python"] is True
    assert payload["backends"]["rust"]["event_count_matches_python"] is True
    assert set(payload["measured_order"]) == {"python", "rust", "mojo"}


def test_source_hashes_cover_the_declared_implementation_surfaces() -> None:
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
    """Require an explicit acknowledgement for multi-CPU affinity."""
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


def test_explicit_partial_run_records_unavailable_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep unavailable-backend evidence structured when explicitly requested."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python", "mojo"))
    monkeypatch.setattr(benchmark, "N_STEPS", 1)
    monkeypatch.setattr(benchmark, "N_REPEATS", 1)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(
        benchmark,
        "_probe_backend",
        lambda backend: (backend == "python", "missing Mojo" if backend == "mojo" else ""),
    )
    output = tmp_path / "partial.json"

    assert benchmark.main(["--json", str(output), "--allow-unavailable-backends"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backends"]["mojo"] == {
        "available": False,
        "unavailable_reason": "missing Mojo",
        "used": False,
    }


@pytest.mark.parametrize(
    ("mojo_trace", "mojo_spikes", "expected"),
    [(np.array([0.0]), 1, 3), (np.array([3.0e-6]), 0, 4)],
)
def test_parity_contract_failures_have_distinct_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mojo_trace: npt.NDArray[np.float64],
    mojo_spikes: int,
    expected: int,
) -> None:
    """Distinguish event failure from bounded-state failure."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python", "mojo"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})

    def measured(
        backend: str,
    ) -> tuple[float, float, npt.NDArray[np.float64], int, tuple[float, ...]]:
        if backend == "python":
            return 1.0, 1.0, np.array([0.0]), 0, (0.0,) * 6
        return 0.5, 0.5, mojo_trace, mojo_spikes, (0.0,) * 6

    monkeypatch.setattr(benchmark, "_measure_backend", measured)
    assert benchmark.main(["--json", str(tmp_path / f"failure-{expected}.json")]) == expected


def test_optional_host_metadata_and_invalid_tool_are_fail_soft(tmp_path: Path) -> None:
    """Retain report generation when optional host metadata is unavailable."""
    metadata = tmp_path / "metadata"
    metadata.write_text(" powersave\n", encoding="utf-8")
    assert benchmark._read_optional(metadata) == "powersave"
    assert benchmark._read_optional(tmp_path / "missing") == "unavailable"
    assert benchmark._tool_version([]) == "unavailable"
    assert benchmark._tool_version([str(tmp_path / "not-an-executable")]) == "unavailable"


def test_cpu_and_tool_path_portable_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover hosts without Linux CPU metadata or PATH-installed runtimes."""

    def read_text(_path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        del encoding, errors
        raise OSError("cpuinfo unavailable")

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(platform, "processor", lambda: "portable-cpu")
    assert benchmark._cpu_model() == "portable-cpu"

    runtime = tmp_path / "mojo"
    runtime.touch()
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert benchmark._tool_path("mojo", runtime) == str(runtime)
    assert benchmark._tool_path("missing", tmp_path / "absent") is None


def test_backend_order_requires_python_reference_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed if a future backend ordering bypasses the golden trace."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("mojo", "python"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(
        benchmark,
        "_measure_backend",
        lambda _backend: (1.0, 1.0, np.array([0.0]), 0, (0.0,) * 6),
    )

    with pytest.raises(RuntimeError, match="Python reference must be measured first"):
        benchmark.main(["--json", str(tmp_path / "invalid-order.json")])
