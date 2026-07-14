# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation-to-Verilog benchmark evidence tests

"""Validate committed compiler evidence and the real reduced benchmark CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

from benchmarks import bench_verilog_compiler as benchmark


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks/results/bench_verilog_compiler.json"
EXPECTED_CASES = {
    "euler_registered",
    "rk4_registered",
    "substep_rk4_registered",
    "escape_rate_registered",
    "euler_folded_parameter_port",
    "sqrt_map_registered",
    "nearest_half_map_registered",
}


def _mapping(value: object) -> dict[str, object]:
    """Narrow a known JSON mapping at an evidence boundary."""
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _payload(path: Path = ARTIFACT) -> dict[str, object]:
    """Load one compiler benchmark payload."""
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    """Return a source-file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_evidence_discloses_loaded_host_scope() -> None:
    """The artifact cannot be mistaken for isolated production evidence."""
    payload = _payload()
    isolation = _mapping(payload["isolation"])
    workload = _mapping(payload["workload"])

    assert payload["schema_version"] == 1
    assert payload["evidence_class"] == "local_regression"
    assert payload["working_directory"] == "repository_root"
    assert isolation["classification"] == "loaded_host"
    assert isolation["exclusive_core_reserved"] is False
    assert isolation["other_heavy_jobs_running"] in {"yes", "no", "unknown"}
    assert isolation["other_heavy_jobs_note"] != "not disclosed"
    assert workload["sampling"] == "round_robin_rotating_start"
    assert workload["timed_region"] == "compiler_call_only"
    assert "not promotion-grade" in cast(str, payload["interpretation"])


def test_committed_evidence_binds_every_compiler_source() -> None:
    """Every recorded source digest matches the live repository file."""
    hashes = _mapping(_payload()["source_sha256"])

    assert set(hashes) == set(benchmark.SOURCE_PATHS)
    for relative_path, digest in hashes.items():
        assert digest == _sha256(ROOT / relative_path)


def test_committed_cases_have_complete_deterministic_samples() -> None:
    """All seven public workloads retain timings and stable output identities."""
    payload = _payload()
    cases = _mapping(payload["cases"])
    workload = _mapping(payload["workload"])
    samples = cast(int, workload["samples_per_case"])

    assert set(cases) == EXPECTED_CASES
    for raw_row in cases.values():
        row = _mapping(raw_row)
        assert len(cast(list[int], row["samples_ns"])) == samples
        assert cast(float, row["minimum_ms"]) <= cast(float, row["median_ms"])
        assert cast(float, row["median_ms"]) <= cast(float, row["maximum_ms"])
        assert len(cast(str, row["output_sha256"])) == 64
        assert cast(int, row["output_lines"]) > 0


def test_committed_scope_rejects_false_polyglot_comparison() -> None:
    """The artifact names Python as authority and makes no stub timing claim."""
    scope = _mapping(_payload()["implementation_scope"])

    assert scope["authority"] == "python_equation_to_verilog_compiler"
    assert scope["cross_language_comparison"] is False
    reason = cast(str, scope["reason"])
    for language in ("Go", "Julia", "Mojo", "Rust"):
        assert language in reason
    assert "non-executable" in reason
    assert "generated stubs" in reason


def test_reduced_cli_executes_real_compiler_paths(tmp_path: Path) -> None:
    """The CLI writes valid source-bound evidence without patched compiler calls."""
    output = tmp_path / "compiler.json"
    command = [
        sys.executable,
        "benchmarks/bench_verilog_compiler.py",
        "--samples",
        "2",
        "--warmup",
        "1",
        "--json",
        str(output),
        "--other-heavy-jobs-running",
        "unknown",
        "--other-heavy-jobs-note",
        "focused test execution",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = _payload(output)

    assert completed.returncode == 0
    assert set(_mapping(payload["cases"])) == EXPECTED_CASES
    assert _mapping(payload["workload"])["samples_per_case"] == 2
    assert set(_mapping(payload["source_sha256"])) == set(benchmark.SOURCE_PATHS)
