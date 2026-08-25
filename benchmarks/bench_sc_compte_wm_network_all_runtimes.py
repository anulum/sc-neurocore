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
import statistics
import subprocess
import time
from typing import Any

from sc_neurocore.network import (
    SCCompteWMBackend,
    run_sc_compte_wm_network,
    sc_compte_wm_backend_status,
)

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
    "src/sc_neurocore/accel/mojo/sc_compte_wm_network/libsc_compte_wm_network.so",
    "src/sc_neurocore/accel/mojo/sc_compte_wm_network/__init__.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(command: list[str]) -> str:
    executed = subprocess.run(
        command,
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (executed.stdout or executed.stderr).strip()


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "rust": _version([str(REPOSITORY / ".venv/bin/rustc"), "--version"]),
        "julia": _version([str(REPOSITORY / ".venv/bin/julia"), "--version"]),
        "go": _version([str(REPOSITORY / ".venv/bin/go"), "version"]),
        "mojo": _version([str(REPOSITORY / ".venv/bin/mojo"), "--version"]),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
    }


def build_payload(steps: int, repeats: int) -> dict[str, Any]:
    """Execute every selected runtime and return exact parity evidence."""
    if steps <= 0 or repeats <= 0:
        raise ValueError("steps and repeats must be positive")
    statuses = {status.backend: status for status in sc_compte_wm_backend_status()}
    unavailable = [backend for backend in BACKENDS if not statuses[backend].available]
    if unavailable:
        raise RuntimeError("unavailable SC Compte backends: " + ", ".join(unavailable))
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
        "environment": _environment(),
        "source_sha256": {relative: _sha256(REPOSITORY / relative) for relative in SOURCE_PATHS},
        "backends": results,
        "all_backends_available": not unavailable,
        "all_runtime_input_spike_count_exact": exact_custody,
        "all_repeat_receipts_exact": repeats_exact,
        "passed": not unavailable and exact_custody and repeats_exact,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_payload(args.steps, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
