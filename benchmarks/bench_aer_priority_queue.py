# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - AER priority queue benchmark and evidence writer.

"""Benchmark and invariant evidence for the AER priority queue.

The timing fields are local regression evidence. Production throughput claims
require a rerun on isolated cores, with the recorded affinity/load/governor
fields used to prove isolation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.hdl import AERPriorityEvent, AERPriorityQueueReference  # noqa: E402


DEFAULT_JSON = (
    REPO_ROOT / "benchmarks" / "results" / "local_python_2026-06-04_aer_priority_queue.json"
)


def _host_context() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    frequency_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    governor = (
        governor_path.read_text(encoding="utf-8").strip() if governor_path.exists() else "unknown"
    )
    frequency_khz = (
        int(frequency_path.read_text(encoding="utf-8").strip()) if frequency_path.exists() else None
    )
    kernel_isolated_path = Path("/sys/devices/system/cpu/isolated")
    nohz_full_path = Path("/sys/devices/system/cpu/nohz_full")
    cgroup_path = _current_cgroup_path()
    cgroup_cpuset = _read_cgroup_value(cgroup_path, "cpuset.cpus")
    cgroup_effective_cpuset = _read_cgroup_value(cgroup_path, "cpuset.cpus.effective")
    return {
        "cpu_affinity": affinity,
        "cgroup_path": cgroup_path,
        "cgroup_cpuset": cgroup_cpuset,
        "cgroup_effective_cpuset": cgroup_effective_cpuset,
        "kernel_isolated_cpus": (
            kernel_isolated_path.read_text(encoding="utf-8").strip()
            if kernel_isolated_path.exists()
            else ""
        ),
        "kernel_nohz_full_cpus": (
            nohz_full_path.read_text(encoding="utf-8").strip() if nohz_full_path.exists() else ""
        ),
        "load_average_before": list(os.getloadavg()),
        "cpu0_governor": governor,
        "cpu0_frequency_khz": frequency_khz,
        "runtime_cpuset_shield_claimed": bool(cgroup_effective_cpuset)
        and cgroup_effective_cpuset != "0-11",
        "kernel_isolated_core_claimed": (
            kernel_isolated_path.exists()
            and bool(kernel_isolated_path.read_text(encoding="utf-8").strip())
        ),
    }


def _current_cgroup_path() -> str:
    proc_cgroup = Path("/proc/self/cgroup")
    if not proc_cgroup.exists():
        return ""
    for line in proc_cgroup.read_text(encoding="utf-8").splitlines():
        fields = line.split(":", maxsplit=2)
        if len(fields) == 3 and fields[0] == "0":
            return fields[2].lstrip("/")
    return ""


def _read_cgroup_value(cgroup_path: str, name: str) -> str:
    if not cgroup_path:
        return ""
    path = Path("/sys/fs/cgroup") / cgroup_path / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _make_events(n_events: int) -> list[AERPriorityEvent]:
    return [
        AERPriorityEvent(
            target_id=(index * 17) & 0xFF,
            weight=((index * 31) % 255) - 127,
            timestamp=index & 0xFFFF,
            priority=(index * 5 + (index // 7)) & 0x3,
        )
        for index in range(n_events)
    ]


def _run_reference(events: list[AERPriorityEvent], repeats: int) -> tuple[float, dict[str, int]]:
    timings: list[float] = []
    invariant_totals = {
        "priority_violations": 0,
        "fifo_tie_violations": 0,
        "backpressure_rejections": 0,
        "deadline_traps": 0,
    }

    for _ in range(repeats):
        queue = AERPriorityQueueReference(capacity=64, max_latency_cycles=3)
        t0 = time.perf_counter_ns()
        for cycle, event in enumerate(events):
            out_ready = cycle % 11 != 0
            step = queue.step(event, out_ready=out_ready)
        while not queue.empty:
            queue.step(None, out_ready=True)
        t1 = time.perf_counter_ns()
        timings.append((t1 - t0) / len(events))

        ordering_queue = AERPriorityQueueReference(capacity=len(events))
        ordering_queue.extend(events)
        emitted = ordering_queue.drain()
        priorities = [event.priority for event in emitted]
        invariant_totals["priority_violations"] += sum(
            1 for left, right in zip(priorities, priorities[1:]) if left > right
        )
        for priority in range(4):
            original = [event.timestamp for event in events if event.priority == priority]
            observed = [event.timestamp for event in emitted if event.priority == priority]
            if observed != original[: len(observed)]:
                invariant_totals["fifo_tie_violations"] += 1
        invariant_totals["backpressure_rejections"] += int(queue.dropped_event)

        trap_queue = AERPriorityQueueReference(capacity=4, max_latency_cycles=3)
        trap_queue.enqueue(AERPriorityEvent(target_id=0, weight=0, timestamp=0, priority=0))
        for _cycle in range(5):
            trap_queue.step(None, out_ready=False)
        invariant_totals["deadline_traps"] += int(trap_queue.critical_deadline_violation)

    return statistics.median(timings), invariant_totals


def _run_yosys() -> dict[str, Any]:
    yosys = shutil.which("yosys")
    if yosys is None:
        return {
            "available": False,
            "exit_code": -1,
            "elapsed_ns": 0,
            "cell_count": None,
        }

    command = [
        yosys,
        "-p",
        "read_verilog -sv hdl/sc_aer_priority_queue.v; synth -top sc_aer_priority_queue; stat",
    ]
    t0 = time.perf_counter_ns()
    proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    t1 = time.perf_counter_ns()
    cell_count: int | None = None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Number of cells:"):
            cell_count = int(stripped.rsplit(" ", maxsplit=1)[-1])
    return {
        "available": True,
        "exit_code": proc.returncode,
        "elapsed_ns": t1 - t0,
        "cell_count": cell_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--events", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if args.events <= 0 or args.repeats <= 0:
        raise SystemExit("--events and --repeats must be positive")

    host_context = _host_context()
    events = _make_events(args.events)
    median_ns_per_event, invariants = _run_reference(events, args.repeats)
    yosys_report = _run_yosys()
    host_context["load_average_after"] = list(os.getloadavg())

    payload = {
        "benchmark": "aer_priority_queue_backpressure_contract",
        "language": "Python+SystemVerilog",
        "events": args.events,
        "repeats": args.repeats,
        "median_ns_per_event": median_ns_per_event,
        "priority_violations": invariants["priority_violations"],
        "fifo_tie_violations": invariants["fifo_tie_violations"],
        "backpressure_rejections": invariants["backpressure_rejections"],
        "deadline_traps": invariants["deadline_traps"],
        "yosys": yosys_report,
        "host_context": host_context,
        "hardware_measurement_claimed": False,
        "benchmark_isolation_mode": "runtime-cpuset-shield",
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
