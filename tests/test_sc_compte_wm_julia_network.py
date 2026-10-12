# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Custody tests for the documented repository-local Julia network lane."""

from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from tests.toolchain_support import require_executable
from tests.benchmark_history_support import assert_committed_source_hashes

REPOSITORY = Path(__file__).resolve().parents[1]
JULIA_ROOT = REPOSITORY / "src/sc_neurocore/accel/julia/sc_compte_wm_network"
PROJECT = JULIA_ROOT / "Project.toml"
MANIFEST = JULIA_ROOT / "Manifest.toml"
SOURCE = JULIA_ROOT / "SCCompteWMNetwork.jl"
NATIVE_TEST = JULIA_ROOT / "test_sc_compte_wm_network.jl"
BENCHMARK = JULIA_ROOT / "bench_sc_compte_wm_network.jl"
RESULT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network_julia.toml"
JULIA = require_executable("julia")
NETWORK_STACK = frozenset(
    {
        "Downloads",
        "FFTW",
        "LazyArtifacts",
        "LibCURL",
        "LibCURL_jll",
        "LibGit2",
        "LibGit2_jll",
        "LibSSH2_jll",
        "MKL_jll",
        "MbedTLS_jll",
        "Pkg",
        "Zlib_jll",
        "nghttp2_jll",
        "p7zip_jll",
    }
)


def _environment() -> dict[str, str]:
    """Return the process environment with the repository-local Julia depot."""
    environment = os.environ.copy()
    environment["JULIA_DEPOT_PATH"] = str(REPOSITORY / ".venv/julia_depot")
    return environment


def test_julia_network_native_parity_suite() -> None:
    """Run the native Julia suite of the lane with the locked project and require it to pass."""
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
    """Bind the declared dependencies, the narrow lock and the documented public symbols."""
    project = tomllib.loads(PROJECT.read_text(encoding="utf-8"))
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    assert project["deps"] == {
        "FFTW_jll": "f5851436-0d7a-5f13-b9de-f02708fd171a",
        "SHA": "ea8e919c-243c-51af-8825-aaa63cd721ce",
    }
    assert manifest["julia_version"] == "1.11.9"
    assert "FFTW_jll" in manifest["deps"]
    # The transform binary is called directly. The lock therefore names no
    # package manager, no downloader and none of their network libraries.
    assert NETWORK_STACK.isdisjoint(manifest["deps"])
    source = SOURCE.read_text(encoding="utf-8")
    assert "using FFTW_jll: libfftw3" in source
    assert "using FFTW\n" not in source
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
    """Require the committed benchmark receipt to carry the digests of its committed sources."""
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
    assert set(payload["source_sha256"]) == {
        path.relative_to(REPOSITORY).as_posix() for path in (PROJECT, MANIFEST, SOURCE, BENCHMARK)
    }
    assert_committed_source_hashes(
        RESULT, repo_root=REPOSITORY, source_hashes=payload["source_sha256"]
    )


def test_julia_1000_step_event_receipt_matches_python() -> None:
    """Compare the Julia input and spike digests of 1,000 steps with the Python receipt."""
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
