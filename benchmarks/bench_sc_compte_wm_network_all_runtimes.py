# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Measure explicit five-runtime SC Compte dispatch and custody parity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

from sc_neurocore.network import (
    SCCompteWMBackend,
    run_sc_compte_wm_network,
    sc_compte_wm_backend_status,
)
from sc_neurocore.accel.mojo.sc_compte_wm_network import LIBRARY_PATH

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network_all_runtimes.json"
BACKENDS: tuple[SCCompteWMBackend, ...] = ("python", "rust", "julia", "go", "mojo")
N_STEPS = 1_000
N_REPEATS = 3
SOURCE_PATHS = (
    "benchmarks/bench_sc_compte_wm_network_all_runtimes.py",
    "src/sc_neurocore/network/sc_compte_wm.py",
    "src/sc_neurocore/network/sc_compte_wm_drive.py",
    "src/sc_neurocore/network/sc_compte_wm_network.py",
    "src/sc_neurocore/network/sc_compte_wm_backends.py",
    "engine/src/sc_compte_wm_network.rs",
    "engine/examples/sc_compte_wm_network_run.rs",
    "src/sc_neurocore/accel/julia/sc_compte_wm_network/SCCompteWMNetwork.jl",
    "src/sc_neurocore/accel/julia/sc_compte_wm_network/run_sc_compte_wm_network.jl",
    "src/sc_neurocore/accel/go/sc_compte_wm_network/network.go",
    "src/sc_neurocore/accel/go/cmd/run_sc_compte_wm_network/main.go",
    "src/sc_neurocore/accel/mojo/sc_compte_wm_network/sc_compte_wm_network.mojo",
    "src/sc_neurocore/accel/mojo/sc_compte_wm_network/__init__.py",
)


def _sha256(path: Path) -> str:
    """Digest the exact source or loaded-library bytes in a recording."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(
    command: list[str], *, cwd: Path = REPOSITORY, env: dict[str, str] | None = None
) -> str:
    """Ask a resolved executable in the directory and environment used by dispatch."""
    executed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (executed.stdout or executed.stderr).strip()


class BenchmarkRuntimeUnavailable(RuntimeError):
    """A required command runtime is absent from the dispatch PATH."""


def _required_tool(name: str) -> str:
    """Resolve a command through PATH, refusing a missing runtime."""
    executable = shutil.which(name)
    if executable is None:
        raise BenchmarkRuntimeUnavailable(f"required {name} runtime is not on PATH")
    return executable


def collect_environment() -> dict[str, object]:
    """Observe dispatch tools and the library used by the in-process Mojo lane.

    Returns
    -------
    dict of str to object
        PATH-resolved commands and versions, Go's module-selected toolchain,
        loaded-library identity and current host context. Absent standalone
        compiler commands are represented by ``None``. Cargo selects the Rust
        compiler, and the Mojo lane loads a library; a current compiler
        observation does not prove that library's builder.

    Raises
    ------
    BenchmarkRuntimeUnavailable
        If a required Rust, Julia or Go command is absent from PATH.
    """
    cargo, julia, go = (_required_tool(name) for name in ("cargo", "julia", "go"))
    rustc = shutil.which("rustc")
    mojo = shutil.which("mojo")
    go_home = REPOSITORY / ".venv/go"
    go_environment = {
        **os.environ,
        "GOTOOLCHAIN": "auto",
        "GOPATH": str(go_home),
        "GOMODCACHE": str(go_home / "pkg/mod"),
        "GOCACHE": str(go_home / "cache"),
    }
    return {
        "python": platform.python_version(),
        "rust": None if rustc is None else _version([rustc, "--version"]),
        "cargo": _version([cargo, "--version"]),
        "julia": _version([julia, "--version"]),
        "go": _version(
            [go, "version"], cwd=REPOSITORY / "src/sc_neurocore/accel/go", env=go_environment
        ),
        "mojo": None if mojo is None else _version([mojo, "--version"]),
        "runtime_commands": {
            "rustc": rustc,
            "cargo": cargo,
            "julia": julia,
            "go": go,
            "mojo": mojo,
        },
        "go_toolchain_selection": "auto in src/sc_neurocore/accel/go",
        "mojo_library": {
            "path": LIBRARY_PATH.relative_to(REPOSITORY).as_posix(),
            "sha256": _sha256(LIBRARY_PATH) if LIBRARY_PATH.is_file() else None,
            "build_compiler_verified": False,
        },
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
    }


def build_payload(steps: int, repeats: int) -> dict[str, Any]:
    """Execute all five runtimes and compare their input, spike and count custody.

    Parameters
    ----------
    steps : int
        Positive number of network steps at the frozen 0.02 ms timestep.
    repeats : int
        Positive number of fresh measured runs per backend, after warmup.

    Returns
    -------
    dict of str to Any
        Source-bound local regression recording with all five routes, actual
        timing samples, repeat determinism and cross-runtime custody checks.

    Raises
    ------
    ValueError
        If steps or repeats are not positive.
    RuntimeError
        If any explicit backend is unavailable or fails; no fallback is used.
    """
    if steps <= 0 or repeats <= 0:
        raise ValueError("steps and repeats must be positive")
    statuses = {status.backend: status for status in sc_compte_wm_backend_status()}
    unavailable = [backend for backend in BACKENDS if not statuses[backend].available]
    if unavailable:
        raise RuntimeError("unavailable SC Compte backends: " + ", ".join(unavailable))
    environment = collect_environment()
    results: dict[str, Any] = {}
    for backend in BACKENDS:
        run_sc_compte_wm_network(
            16 * 0.02,
            backend=backend,
            statistics_window_ms=16 * 0.02,
            timeout_s=600.0,
        )
        execution_samples: list[int] = []
        end_to_end_samples: list[int] = []
        receipts = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            result = run_sc_compte_wm_network(
                steps * 0.02,
                backend=backend,
                statistics_window_ms=500.0,
                timeout_s=600.0,
            )
            end_to_end_samples.append(time.perf_counter_ns() - started)
            execution_samples.append(result.execution_ns)
            receipts.append(result.receipt)
        repeat_exact = len(set(receipts)) == 1
        receipt = receipts[0]
        median_execution = int(statistics.median(execution_samples))
        results[backend] = {
            "execution_mode": statuses[backend].execution_mode,
            "execution_samples_ns": execution_samples,
            "end_to_end_samples_ns": end_to_end_samples,
            "median_execution_ns": median_execution,
            "median_ns_per_network_step": median_execution / steps,
            "input_sha256": receipt.input_sha256,
            "spike_sha256": receipt.spike_sha256,
            "final_state_sha256": receipt.final_state_sha256,
            "spike_counts": {
                "excitatory": receipt.excitatory_spikes,
                "inhibitory": receipt.inhibitory_spikes,
            },
            "repeat_receipts_exact": repeat_exact,
        }
    custody_keys = {
        (
            value["input_sha256"],
            value["spike_sha256"],
            value["spike_counts"]["excitatory"],
            value["spike_counts"]["inhibitory"],
        )
        for value in results.values()
    }
    exact_custody = len(custody_keys) == 1
    repeats_exact = all(value["repeat_receipts_exact"] for value in results.values())
    return {
        "schema_version": "sc-neurocore.sc-compte-wm-all-runtime-benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "SC-COMPTE-WM-NETWORK",
        "evidence_class": "local_regression",
        "production_speed_claimed": False,
        "hardware_measurement_claimed": False,
        "persistent_bump_claimed": False,
        "distractor_resistance_claimed": False,
        "configuration": {
            "cells": 2560,
            "excitatory_cells": 2048,
            "inhibitory_cells": 512,
            "dt_ms": 0.02,
            "steps": steps,
            "duration_ms": steps * 0.02,
            "repeats": repeats,
            "seed": 42,
        },
        "command": [sys.executable, *sys.argv],
        "isolation": {"isolated": False, "method": "shared workstation, no reserved core"},
        "load_average_after": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "environment": environment,
        "source_sha256": {relative: _sha256(REPOSITORY / relative) for relative in SOURCE_PATHS},
        "backends": results,
        "all_backends_available": not unavailable,
        "all_runtime_input_spike_count_exact": exact_custody,
        "all_repeat_receipts_exact": repeats_exact,
        "passed": not unavailable and exact_custody and repeats_exact,
    }


def main() -> int:
    """Write the full recording or print its runtime environment without simulation.

    Returns
    -------
    int
        Zero for an environment report or passing five-runtime recording, one
        for unequal custody. Missing command runtimes produce argparse exit 2.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show-environment", action="store_true")
    args = parser.parse_args()
    try:
        if args.show_environment:
            print(json.dumps(collect_environment(), indent=2))
            return 0
        payload = build_payload(args.steps, args.repeats)
    except BenchmarkRuntimeUnavailable as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
