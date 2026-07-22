# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chialvo benchmark evidence gates

"""Validate Chialvo source binding and fail-closed five-backend evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bench_chialvo_map as benchmark


def _raise_oserror(*_args: object, **_kwargs: object) -> None:
    """Raise the filesystem error used by fail-soft metadata probes."""
    raise OSError("unavailable")


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
    """Bind every declared source path, including the decomposed Rust model module."""
    hashes = benchmark._source_hashes()
    assert set(hashes) == set(benchmark.SOURCE_PATHS)
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected


def test_committed_evidence_is_complete_source_bound_and_within_parity() -> None:
    """Reject stale sources, missing backends, event drift, and unbounded state drift."""
    output = benchmark.REPOSITORY / "benchmarks/results/bench_chialvo_map.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_hashes"] == benchmark._source_hashes()
    assert payload["meta"]["single_cpu_pinned"] is True
    assert set(payload["backends"]) == set(benchmark.BACKENDS)
    for backend in benchmark.BACKENDS:
        row = payload["backends"][backend]
        assert row["available"] is True
        assert row["used"] is True
        assert row["event_count_matches_python"] is True
        assert row["final_state_matches_python"] is True
        assert row["parity_atol"] == benchmark.PARITY_ATOL[backend]
        assert row["parity_max_abs_diff"] <= benchmark.PARITY_ATOL[backend]
        assert row["final_state_max_abs_diff"] <= benchmark.PARITY_ATOL[backend]


def test_unpinned_run_is_rejected_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require one logical CPU unless the diagnostic relaxation is explicit."""
    monkeypatch.setattr(benchmark.os, "sched_getaffinity", lambda _pid: {0, 1})
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output)]) == 2
    assert not output.exists()


def test_missing_backend_is_rejected_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not publish a partial five-backend record as complete evidence."""
    monkeypatch.setattr(benchmark.os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(
        benchmark,
        "_probe_backend",
        lambda backend: (backend != "mojo", "missing") if backend == "mojo" else (True, ""),
    )
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output)]) == 2
    assert not output.exists()


def test_explicit_partial_backend_run_is_labelled_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow an explicitly requested diagnostic record without calling the missing lane."""
    monkeypatch.setattr(benchmark.os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(
        benchmark,
        "_probe_backend",
        lambda backend: (backend != "mojo", "missing") if backend == "mojo" else (True, ""),
    )
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(
        benchmark,
        "_measure_backend",
        lambda _backend: (1.0, 1.0, np.zeros(4), 1, 0.0, 0.0),
    )
    output = tmp_path / "diagnostic.json"
    assert benchmark.main(["--json", str(output), "--allow-unavailable-backends"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backends"]["mojo"] == {
        "available": False,
        "unavailable_reason": "missing",
        "used": False,
    }


def test_metadata_helpers_report_unavailable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover fail-soft host metadata without weakening the fidelity gates."""
    monkeypatch.setattr(Path, "read_text", _raise_oserror)
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "")
    assert benchmark._cpu_model() == "unknown"
    assert benchmark._read_optional(tmp_path / "missing") == "unavailable"
    assert benchmark._tool_version([]) == "unavailable"
    monkeypatch.setattr(benchmark.subprocess, "run", _raise_oserror)
    assert benchmark._tool_version(["missing"]) == "unavailable"


def test_metadata_helpers_cover_fallback_and_empty_tool_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve explicit tool fallbacks and stable no-output version records."""
    fallback = tmp_path / "tool"
    fallback.write_text("tool", encoding="utf-8")
    monkeypatch.setattr(benchmark.shutil, "which", lambda _name: None)
    assert benchmark._tool_path("missing", fallback) == str(fallback)
    assert benchmark._tool_path("missing") is None
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: "processor: generic")
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "fallback-cpu")
    assert benchmark._cpu_model() == "fallback-cpu"
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="", stderr="", returncode=7),
    )
    assert benchmark._tool_version(["tool"]) == "exit 7"


def test_native_measurement_before_python_reference_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the Python-reference-first invariant explicit and fail closed."""
    monkeypatch.setattr(benchmark.os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(benchmark, "BACKENDS", ("rust", "python"))
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(
        benchmark,
        "_measure_backend",
        lambda _backend: (1.0, 1.0, np.zeros(4), 1, 0.0, 0.0),
    )
    output = tmp_path / "unused.json"
    with pytest.raises(RuntimeError, match="Python reference must be measured first"):
        benchmark.main(["--json", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("failure", ["trace", "final_state", "event_count"])
def test_fidelity_failure_is_not_published(
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Withhold evidence when any maintained fidelity observable exceeds its contract."""
    monkeypatch.setattr(benchmark.os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(benchmark.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(benchmark, "_probe_backend", lambda _backend: (True, ""))
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})

    def measured(backend: str) -> tuple[float, float, np.ndarray, int, float, float]:
        is_failing_backend = backend == "mojo"
        trace_value = 2.0e-6 if is_failing_backend and failure == "trace" else 0.0
        event_count = 2 if is_failing_backend and failure == "event_count" else 1
        x_final = 2.0e-6 if is_failing_backend and failure == "final_state" else 0.0
        return 1.0, 1.0, np.full(4, trace_value), event_count, x_final, 0.0

    monkeypatch.setattr(benchmark, "_measure_backend", measured)
    output = tmp_path / "unused.json"
    assert benchmark.main(["--json", str(output)]) == 3
    assert not output.exists()
