#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - UltraScale+ target benchmark evidence

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from gen_vivado_project import generate_tcl, load_manifest, sku_baseline


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


def _write_fixture(root: Path, sku: str) -> Path:
    source = root / "top.sv"
    source.write_text(
        "module top(input wire clk, input wire rst_n, output wire out); assign out = clk & rst_n; endmodule\n",
        encoding="utf-8",
    )
    xdc = root / f"{sku}.xdc"
    xdc.write_text(
        "create_clock -name sc_neurocore_clk -period 4.000 [get_ports clk]\n",
        encoding="utf-8",
    )
    manifest = root / f"{sku}.json"
    manifest.write_text(
        json.dumps(
            {
                "top": "top",
                "sku": sku,
                "clock_mhz": 250,
                "sources": ["top.sv"],
                "xdc": [f"{sku}.xdc"],
                "output_dir": f"out/{sku}",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def run(iterations: int, repeats: int) -> dict[str, Any]:
    baseline = sku_baseline()
    medians: list[float] = []
    checksum = 0
    with tempfile.TemporaryDirectory(prefix="sc_neurocore_usplus_") as tmp_raw:
        tmp = Path(tmp_raw)
        manifests = [_write_fixture(tmp, "zu3eg"), _write_fixture(tmp, "zu9eg")]
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                for manifest_path in manifests:
                    manifest = load_manifest(manifest_path)
                    tcl = generate_tcl(manifest)
                    checksum ^= len(tcl)
                    if "DSP" + "58" in tcl:
                        raise AssertionError("UltraScale+ Tcl must not claim a newer-family DSP mapping")
            elapsed = time.perf_counter_ns() - start
            medians.append(elapsed / max(iterations * len(manifests), 1))
    ordered = sorted(medians)
    return {
        "benchmark": "ultrascale_plus_target_contract",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language": "Python",
        "language_surfaces": ["Python", "Vivado Tcl"],
        "benchmark_isolation_mode": "runtime-cpuset-shield",
        "hardware_measurement_claimed": False,
        "iterations": iterations,
        "repeats": repeats,
        "manifest_count": 2,
        "median_ns_per_manifest": ordered[len(ordered) // 2],
        "min_ns_per_manifest": ordered[0],
        "max_ns_per_manifest": ordered[-1],
        "checksum": checksum,
        "sku_baseline": baseline,
        "host_context": _host_context(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark UltraScale+ target project generation.")
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.iterations, args.repeats)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
