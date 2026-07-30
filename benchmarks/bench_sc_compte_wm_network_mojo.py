# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

"""Measure complete native Mojo SC Compte network steps without behavior claims."""

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

from sc_neurocore.accel.mojo.sc_compte_wm_network import SCCompteWMMojoNetwork

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network_mojo.json"
N_STEPS = 1_000
N_REPEATS = 3
SOURCE_PATHS = (
    "benchmarks/bench_sc_compte_wm_network_mojo.py",
    "src/sc_neurocore/accel/mojo/sc_compte_wm_network/__init__.py",
    "src/sc_neurocore/accel/mojo/sc_compte_wm_network/sc_compte_wm_network.mojo",
    "src/sc_neurocore/accel/mojo/sc_compte_wm_network/libsc_compte_wm_network.so",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mojo_version() -> str:
    executed = subprocess.run(
        [str(REPOSITORY / ".venv/bin/mojo"), "--version"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executed.stdout.strip()


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "mojo": _mojo_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
    }


def build_payload(steps: int, repeats: int) -> dict[str, Any]:
    """Measure fresh native runs and return source/binary-bound evidence."""
    if steps <= 0 or repeats <= 0:
        raise ValueError("steps and repeats must be positive")
    # Construct spectra and warm the shared-library boundary outside timing.
    SCCompteWMMojoNetwork().run(16 * 0.02, statistics_window_ms=16 * 0.02)
    samples_ns: list[int] = []
    input_digests: list[str] = []
    spike_digests: list[str] = []
    state_digests: list[str] = []
    spike_counts: list[tuple[int, int]] = []
    for _ in range(repeats):
        network = SCCompteWMMojoNetwork()
        started = time.perf_counter_ns()
        receipt = network.run(steps * network.spec.dt_ms, statistics_window_ms=500.0)
        samples_ns.append(time.perf_counter_ns() - started)
        input_digests.append(receipt.input_sha256)
        spike_digests.append(receipt.spike_sha256)
        state_digests.append(receipt.final_state_sha256)
        spike_counts.append((receipt.excitatory_spikes, receipt.inhibitory_spikes))
    deterministic = (
        len(set(input_digests))
        == len(set(spike_digests))
        == len(set(state_digests))
        == len(set(spike_counts))
        == 1
    )
    median_ns = int(statistics.median(samples_ns))
    return {
        "schema_version": "sc-neurocore.sc-compte-wm-network-benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "SC-COMPTE-WM-NETWORK",
        "execution_path": "mojo-midpoint-rk2-radix2-fft-x86-64-v3",
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
            "target_cpu": "x86-64-v3",
        },
        "environment": _environment(),
        "source_sha256": {relative: _sha256(REPOSITORY / relative) for relative in SOURCE_PATHS},
        "samples_ns": samples_ns,
        "median_ns": median_ns,
        "median_ns_per_network_step": median_ns / steps,
        "median_cell_updates_per_second": 2560 * steps / (median_ns / 1.0e9),
        "input_sha256": input_digests[0],
        "spike_sha256": spike_digests[0],
        "final_state_sha256": state_digests[0],
        "spike_counts": {
            "excitatory": spike_counts[0][0],
            "inhibitory": spike_counts[0][1],
        },
        "repeat_receipts_exact": deterministic,
        "passed": deterministic,
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
