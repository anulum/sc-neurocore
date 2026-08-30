# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ExpIF five-backend benchmark contract tests

"""Production-path tests for the controlled ExpIF benchmark harness."""

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

import sc_neurocore.neurons.models.expif as expif
from benchmarks import bench_model_expif as benchmark


def test_real_benchmark_writes_complete_parity_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every measured backend through the real simulation API."""
    monkeypatch.setattr(benchmark, "N_STEPS", 3)
    monkeypatch.setattr(benchmark, "N_REPEATS", 1)
    output = tmp_path / "expif.json"

    assert benchmark.main(["--json", str(output), "--allow-unpinned"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == "expif_fourcaud_trocme_2003_complete"
    assert payload["workload"]["n_steps"] == 3
    assert set(payload["measured_order"]) == {"python", "rust", "julia", "go", "mojo"}
    assert all(row["events_match_python"] for row in payload["backends"].values())
    assert all(
        {"event_sha256", "voltage_sha256", "refractory_sha256"} <= row.keys()
        for row in payload["backends"].values()
    )
    assert set(payload["backends"]["python"]["final_state"]) == {
        "v",
        "refractory_remaining",
    }


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
    ("compiled_trace", "compiled_events", "expected"),
    [
        (np.array([0.0]), np.array([1], dtype=np.uint8), 3),
        (np.array([6.0e-8]), np.array([0], dtype=np.uint8), 4),
    ],
)
def test_parity_contract_failures_have_distinct_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_trace: npt.NDArray[np.float64],
    compiled_events: npt.NDArray[np.uint8],
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
    ) -> tuple[
        float,
        float,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.uint8],
        tuple[float, float],
    ]:
        if backend == "python":
            return (
                1.0,
                1.0,
                np.array([0.0]),
                np.array([0.0]),
                np.array([0], dtype=np.uint8),
                (0.0, 0.0),
            )
        return (
            0.5,
            0.5,
            compiled_trace,
            np.array([0.0]),
            compiled_events,
            (0.0, 0.0),
        )

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


def test_backend_probes_report_each_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind every backend name to its real availability probe."""
    monkeypatch.setattr(expif, "_HAS_RUST", True)
    monkeypatch.setattr(expif, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_go_loaded", lambda: True)
    monkeypatch.setattr(expif, "_ensure_mojo_loaded", lambda: False)
    assert benchmark._probe_backend("python") == (True, "")
    assert benchmark._probe_backend("rust") == (True, "")
    assert benchmark._probe_backend("julia")[0] is False
    assert benchmark._probe_backend("go") == (True, "")
    assert benchmark._probe_backend("mojo")[0] is False


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
        lambda _backend: (
            1.0,
            1.0,
            np.array([0.0]),
            np.array([0.0]),
            np.array([0], dtype=np.uint8),
            (0.0, 0.0),
        ),
    )

    with pytest.raises(RuntimeError, match="Python reference must be measured first"):
        benchmark.main(["--json", str(tmp_path / "invalid-order.json")])
