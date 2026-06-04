#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - UltraScale+ dense-folding benchmark evidence

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ultrascale_dense_folding import plan_dense_fold


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _cpuset() -> str | None:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("Cpus_allowed_list:"):
                return line.split(":", 1)[1].strip()
    return _read_text(Path("/sys/fs/cgroup/cpuset.cpus.effective"))


def _host_context() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    cpuset = _cpuset()
    return {
        "affinity_cpus": affinity,
        "affinity_cpu_count": len(affinity),
        "cgroup_effective_cpuset": cpuset,
        "load_average": list(os.getloadavg()),
        "runtime_cpuset_shield_claimed": cpuset == "10-11" or affinity == [10, 11],
    }


def _run_yosys() -> dict[str, Any]:
    yosys = shutil.which("yosys")
    if yosys is None:
        raise RuntimeError("yosys is required for dense-folded HDL evidence")
    script = (
        "read_verilog -sv hdl/sc_dense_folded_q88_core.v; "
        "chparam -set N_INPUTS 8 -set N_NEURONS 8 -set PARALLEL_NEURONS 4 sc_dense_folded_q88_core; "
        "hierarchy -top sc_dense_folded_q88_core; proc; opt; stat -json"
    )
    start = time.perf_counter_ns()
    completed = subprocess.run(
        [yosys, "-p", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed_ns = time.perf_counter_ns() - start
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    cells = 0
    for line in completed.stdout.splitlines():
        if "\"num_cells\":" in line:
            cells = int(line.split(":", 1)[1].strip().rstrip(","))
            break
    return {
        "exit_code": completed.returncode,
        "elapsed_ns": elapsed_ns,
        "cells": cells,
        "parameterisation": "N_INPUTS=8,N_NEURONS=8,PARALLEL_NEURONS=4",
    }


def run(iterations: int, repeats: int) -> dict[str, Any]:
    medians: list[float] = []
    checksum = 0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            plan = plan_dense_fold(n_inputs=64, n_outputs=32, dsp_budget=360)
            checksum ^= plan.dsp_per_cycle ^ plan.compute_cycles
        elapsed = time.perf_counter_ns() - start
        medians.append(elapsed / max(iterations, 1))
    ordered = sorted(medians)
    plan = plan_dense_fold(n_inputs=64, n_outputs=32, dsp_budget=360)
    return {
        "benchmark": "ultrascale_plus_dense_folding_contract",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language": "Python+SystemVerilog",
        "benchmark_isolation_mode": "runtime-cpuset-shield",
        "hardware_measurement_claimed": False,
        "iterations": iterations,
        "repeats": repeats,
        "median_ns_per_plan": ordered[len(ordered) // 2],
        "min_ns_per_plan": ordered[0],
        "max_ns_per_plan": ordered[-1],
        "checksum": checksum,
        "plan": plan.to_dict(),
        "yosys": _run_yosys(),
        "host_context": _host_context(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark UltraScale+ dense folding planner.")
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.iterations, args.repeats)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
