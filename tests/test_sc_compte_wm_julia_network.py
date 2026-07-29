# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

"""Custody tests for the documented repository-local Julia network lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import subprocess
import tomllib

REPOSITORY = Path(__file__).resolve().parents[1]
JULIA_ROOT = REPOSITORY / "src/sc_neurocore/accel/julia/sc_compte_wm_network"
PROJECT = JULIA_ROOT / "Project.toml"
MANIFEST = JULIA_ROOT / "Manifest.toml"
SOURCE = JULIA_ROOT / "SCCompteWMNetwork.jl"
NATIVE_TEST = JULIA_ROOT / "test_sc_compte_wm_network.jl"
BENCHMARK = JULIA_ROOT / "bench_sc_compte_wm_network.jl"
RESULT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network_julia.toml"
JULIA = Path("/home/anulum/.juliaup/bin/julia")


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["JULIA_DEPOT_PATH"] = str(REPOSITORY / ".venv/julia_depot")
    return environment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_julia_network_native_parity_suite() -> None:
    executed = subprocess.run(
        [str(JULIA), f"--project={PROJECT.parent}", str(NATIVE_TEST)],
        cwd=REPOSITORY,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "SC Compte Julia network" in executed.stdout
    assert "Pass" in executed.stdout


def test_julia_dependency_and_native_docs_are_explicit() -> None:
    project = tomllib.loads(PROJECT.read_text(encoding="utf-8"))
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    assert project["deps"]["FFTW"] == "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    assert "FFTW" in manifest["deps"]
    source = SOURCE.read_text(encoding="utf-8")
    for symbol in (
        "SCCompteWMNetworkSpec",
        "SCCompteWMNetworkState",
        "SCCompteWMNetworkRuntime",
        "counter_poisson_counts",
        "step_with_events!",
        "run!",
    ):
        offset = source.index(symbol)
        assert '"""' in source[max(0, offset - 900) : offset]


def test_julia_benchmark_receipt_is_source_bound() -> None:
    payload = tomllib.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["model"] == "SC-COMPTE-WM-NETWORK"
    assert payload["execution_path"] == "julia-midpoint-rk2-fftw"
    assert payload["configuration"]["cells"] == 2560
    assert payload["configuration"]["steps"] == 1000
    assert payload["configuration"]["repeats"] == 3
    assert payload["repeat_receipts_exact"] is True
    assert payload["passed"] is True
    assert payload["persistent_bump_claimed"] is False
    assert payload["distractor_resistance_claimed"] is False
    source_hashes = payload["source_sha256"]
    for path in (PROJECT, MANIFEST, SOURCE, BENCHMARK):
        relative = path.relative_to(REPOSITORY).as_posix()
        assert source_hashes[relative] == _sha256(path)


def test_julia_1000_step_event_receipt_matches_python() -> None:
    julia = tomllib.loads(RESULT.read_text(encoding="utf-8"))
    python = json.loads(
        (REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network.json").read_text(
            encoding="utf-8"
        )
    )
    assert julia["configuration"]["steps"] == python["configuration"]["steps"] == 1000
    assert julia["input_sha256"] == python["input_sha256"]
    assert julia["spike_sha256"] == python["spike_sha256"]
    assert julia["spike_counts"] == python["spike_counts"]
