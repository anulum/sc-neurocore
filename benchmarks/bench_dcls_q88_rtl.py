#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - DCLS Q8.8 RTL benchmark evidence

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

from cosim_dcls_q88_vs_pytorch import dcls_q88_reference, run_deterministic_suite

HDL_FILES = [
    REPO_ROOT / "hdl" / "sc_dcls_axonal_delay.v",
    REPO_ROOT / "hdl" / "sc_dcls_tent_kernel.v",
    REPO_ROOT / "hdl" / "sc_dcls_layer_core.v",
]
FORMAL = REPO_ROOT / "hdl" / "formal" / "sc_dcls_layer_core.sby"


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


def _governors(affinity: list[int]) -> dict[str, str]:
    values: dict[str, str] = {}
    for cpu in affinity[:8]:
        value = _read_text(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"))
        if value is not None:
            values[str(cpu)] = value
    return values


def _host_context() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    cpuset = _cpuset()
    return {
        "affinity_cpus": affinity,
        "affinity_cpu_count": len(affinity),
        "cgroup_effective_cpuset": cpuset,
        "load_average": list(os.getloadavg()),
        "cpu_governors_sample": _governors(affinity),
        "runtime_cpuset_shield_claimed": cpuset == "10-11" or affinity == [10, 11],
    }


def _run_formal() -> dict[str, Any]:
    if shutil.which("sby") is None or shutil.which("cvc5") is None:
        return {"available": False, "exit_code": 127, "elapsed_ns": 0, "passed": False}
    work = REPO_ROOT / "benchmarks" / "results" / ".tmp_dcls_sby"
    shutil.rmtree(work, ignore_errors=True)
    start = time.perf_counter_ns()
    completed = subprocess.run(
        ["sby", "-f", "-d", str(work), str(FORMAL)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter_ns() - start
    shutil.rmtree(work, ignore_errors=True)
    combined = completed.stdout + completed.stderr
    return {
        "available": True,
        "exit_code": completed.returncode,
        "elapsed_ns": elapsed,
        "passed": completed.returncode == 0 and "DONE (PASS" in combined,
    }


def _run_yosys() -> dict[str, Any]:
    if shutil.which("yosys") is None:
        return {"available": False, "exit_code": 127, "elapsed_ns": 0, "cells": -1}
    script = "; ".join(
        [
            "read_verilog -sv " + " ".join(str(path) for path in HDL_FILES),
            "synth -top sc_dcls_layer_core",
            "stat",
        ]
    )
    start = time.perf_counter_ns()
    completed = subprocess.run(
        ["yosys", "-p", script], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    elapsed = time.perf_counter_ns() - start
    cells = -1
    for line in completed.stdout.splitlines():
        if "Number of cells:" in line:
            cells = int(line.rsplit(None, 1)[-1])
    return {
        "available": True,
        "exit_code": completed.returncode,
        "elapsed_ns": elapsed,
        "cells": cells,
    }


def _spike_window(index: int, taps: int) -> list[int]:
    return [1 if ((index + tap * 3) % 7) in (0, 1, 4) else 0 for tap in range(taps)]


def run(samples: int, repeats: int, taps: int) -> dict[str, Any]:
    weights_q88 = [((tap * 73 + 19) % 513) - 256 for tap in range(taps)]
    centre_q88 = min(32767, max(-32768, (taps // 2) << 8))
    sigma_q88 = min(32767, max(256, taps << 8))
    checksum = 0
    overflow_count = 0
    active_tap_total = 0
    start = time.perf_counter_ns()
    for _ in range(repeats):
        for sample in range(samples):
            result = dcls_q88_reference(_spike_window(sample, taps), weights_q88, centre_q88, sigma_q88)
            checksum ^= int(result["accumulator_q16_16"])
            overflow_count += int(bool(result["overflow"]))
            active_tap_total += int(result["active_tap_count"])
    elapsed = time.perf_counter_ns() - start
    cosim = run_deterministic_suite(require_torch=False)
    formal = _run_formal()
    yosys = _run_yosys()
    return {
        "benchmark": "dcls_q88_rtl_contract",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language": "Python+PyTorch+SystemVerilog",
        "language_surfaces": ["Python", "PyTorch", "SystemVerilog"],
        "benchmark_isolation_mode": "runtime-cpuset-shield",
        "hardware_measurement_claimed": False,
        "samples": samples,
        "repeats": repeats,
        "taps": taps,
        "total_samples": samples * repeats,
        "median_ns_per_sample": elapsed / max(samples * repeats, 1),
        "checksum": checksum,
        "overflow_count": overflow_count,
        "active_tap_total": active_tap_total,
        "cosim": cosim,
        "formal": formal,
        "yosys": yosys,
        "host_context": _host_context(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the DCLS Q8.8 reference and RTL evidence.")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--taps", type=int, default=16)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.samples, args.repeats, args.taps)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
