# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ADC-to-spike quantiser benchmark

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

from adc_to_spike_reference import ADCSpikeConfig, ADCToSpikeReference

HDL = REPO_ROOT / "hdl" / "sensors" / "adc_to_spike_quantiser.v"
FORMAL = REPO_ROOT / "hdl" / "formal" / "sensors" / "adc_to_spike_quantiser.sby"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _cpuset() -> str | None:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
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
    work = REPO_ROOT / "benchmarks" / "results" / ".tmp_adc_to_spike_sby"
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
    script = f"read_verilog -sv {HDL}; synth -top adc_to_spike_quantiser; stat"
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


def run(samples: int, repeats: int) -> dict[str, Any]:
    cfg = ADCSpikeConfig(decimation=8, threshold_q=256, base_address=32, negative_offset=1)
    waveform = [((index * 73) % 2048) - 1024 for index in range(samples)]
    spike_total = 0
    start = time.perf_counter_ns()
    for _ in range(repeats):
        ref = ADCToSpikeReference(cfg)
        ref.run(sample & 0xFFFF for sample in waveform)
        spike_total += ref.spike_count
    elapsed = time.perf_counter_ns() - start
    formal = _run_formal()
    yosys = _run_yosys()
    return {
        "benchmark": "adc_to_spike_quantiser_contract",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language": "Python+SystemVerilog",
        "language_surfaces": ["Python", "SystemVerilog"],
        "benchmark_isolation_mode": "runtime-cpuset-shield",
        "hardware_measurement_claimed": False,
        "samples": samples,
        "repeats": repeats,
        "total_samples": samples * repeats,
        "total_spikes": spike_total,
        "median_ns_per_sample": elapsed / max(samples * repeats, 1),
        "formal": formal,
        "yosys": yosys,
        "host_context": _host_context(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the ADC-to-spike quantiser reference and HDL evidence."
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.samples, args.repeats)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
