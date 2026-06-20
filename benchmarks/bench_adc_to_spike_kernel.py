#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-backend benchmark for the ADC-to-spike encoder

"""Measure and compare Python / Rust / Julia / Go / Mojo ADC-to-spike encoders.

The integer per-window ADC-to-spike encoder is exact, so every backend is
bit-identical to the Python floor (parity tolerance zero). The benchmark records,
per backend, ``available`` / ``used`` / ``reason`` plus per-call wall time and
throughput, and writes a JSON artefact under ``benchmarks/results/``.

Host-load and CPU-affinity context is captured per the benchmark-core-isolation
policy. Run pinned for production evidence::

    taskset -c 10-11 python benchmarks/bench_adc_to_spike_kernel.py \\
        --json benchmarks/results/bench_adc_to_spike_kernel.json

An unpinned run on a shared workstation is labelled non-isolated and is valid as
functional / parity / regression evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.sensors.adc_to_spike_kernel import (  # noqa: E402
    ADCSpikeWindowConfig,
    ADCSpikeWindowResult,
    adc_to_spike_windows,
    adc_to_spike_windows_q,
    available_backends,
)

BackendFn = Callable[[npt.ArrayLike, ADCSpikeWindowConfig], ADCSpikeWindowResult]

_RESULT_FIELDS = ("window_values_q", "spike_counts", "polarities")


def _read_text(path: Path) -> str | None:
    """Return a stripped text file, or ``None`` when unreadable."""
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _cpuset() -> str | None:
    """Return the effective cpuset list for the current process."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("Cpus_allowed_list:"):
                return line.split(":", 1)[1].strip()
    return _read_text(Path("/sys/fs/cgroup/cpuset.cpus.effective"))


def _governors(affinity: list[int]) -> dict[str, str]:
    """Sample the CPU frequency governor for the first allowed cores."""
    values: dict[str, str] = {}
    for cpu in affinity[:8]:
        value = _read_text(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"))
        if value is not None:
            values[str(cpu)] = value
    return values


def _tool_version(command: list[str]) -> str | None:
    """Best-effort capture of a backend toolchain version string."""
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return (completed.stdout + completed.stderr).strip().splitlines()[0] if completed else None


def _cpu_model() -> str:
    """Return the host CPU model name."""
    info = _read_text(Path("/proc/cpuinfo"))
    if info is not None:
        for line in info.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _host_context(load_before: list[float], load_after: list[float]) -> dict[str, Any]:
    """Capture CPU affinity, cpuset shield and host load for isolation labelling."""
    affinity = sorted(os.sched_getaffinity(0))
    cpuset = _cpuset()
    shielded = cpuset == "10-11" or affinity == [10, 11]
    return {
        "affinity_cpus": affinity,
        "affinity_cpu_count": len(affinity),
        "cgroup_effective_cpuset": cpuset,
        "load_average_before": load_before,
        "load_average_after": load_after,
        "cpu_governors_sample": _governors(affinity),
        "runtime_cpuset_shield_claimed": shielded,
        "isolation_mode": "runtime-cpuset-shield" if shielded else "non-isolated-shared-host",
    }


def _time_backend(
    fn: BackendFn,
    samples: npt.NDArray[np.int64],
    config: ADCSpikeWindowConfig,
    repeats: int,
) -> tuple[float, ADCSpikeWindowResult]:
    """Warm up once then return the median per-call wall time and a result."""
    result = fn(samples, config)
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(samples, config)
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2], result


def _backend_fn(name: str) -> BackendFn:
    """Return the uniform call wrapper for a backend name."""
    if name == "python":
        return adc_to_spike_windows_q

    def _call(samples: npt.ArrayLike, config: ADCSpikeWindowConfig) -> ADCSpikeWindowResult:
        return adc_to_spike_windows(samples, config, backend=name)

    return _call


def _parity_delta(reference: ADCSpikeWindowResult, candidate: ADCSpikeWindowResult) -> int:
    """Maximum absolute element difference across all result arrays (zero when bit-exact)."""
    delta = 0
    for field in _RESULT_FIELDS:
        ref = np.asarray(getattr(reference, field), dtype=np.int64)
        cand = np.asarray(getattr(candidate, field), dtype=np.int64)
        delta = max(delta, int(np.abs(ref - cand).max()))
    return delta


def run(n_windows: int, seed: int, repeats: int, python_repeats: int) -> dict[str, Any]:
    """Run every available backend and return the benchmark report."""
    config = ADCSpikeWindowConfig()
    rng = np.random.default_rng(seed)
    n_samples = n_windows * config.decimation
    samples = rng.integers(0, 1 << config.adc_width, size=n_samples, dtype=np.int64)
    availability = available_backends()
    load_before = list(os.getloadavg())

    backends: dict[str, dict[str, Any]] = {}
    results: dict[str, ADCSpikeWindowResult] = {}
    order = ("python", "rust", "mojo", "julia", "go")
    for name in order:
        if not availability.get(name, False):
            backends[name] = {"available": False, "used": False, "reason": "backend probe failed"}
            continue
        runs = python_repeats if name == "python" else repeats
        wall, result = _time_backend(_backend_fn(name), samples, config, runs)
        results[name] = result
        backends[name] = {
            "available": True,
            "used": True,
            "reason": "primary reference" if name == "python" else "available",
            "median_call_ms": round(wall * 1e3, 6),
            "samples_per_s": round(n_samples / wall, 1),
            "repeats": runs,
        }

    python_wall = backends["python"]["median_call_ms"]
    for name, info in backends.items():
        if info.get("used") and name != "python":
            info["speedup_over_python"] = round(python_wall / info["median_call_ms"], 2)

    reference_name = "rust" if "rust" in results else "python"
    parity: dict[str, Any] = {"reference": reference_name, "tolerance": 0}
    for name, result in results.items():
        if name == reference_name:
            parity[name] = {"max_abs_delta": 0, "bit_exact": True}
            continue
        delta = _parity_delta(results[reference_name], result)
        parity[name] = {"max_abs_delta": delta, "bit_exact": delta == 0}

    load_after = list(os.getloadavg())
    host_context = _host_context(load_before, load_after)
    return {
        "benchmark": "adc_to_spike_kernel_windows",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language_surfaces": ["Python", "Rust", "Julia", "Go", "Mojo"],
        "kernel": "adc_to_spike_windows_q",
        "hardware_measurement_claimed": False,
        "benchmark_isolation_mode": host_context["isolation_mode"],
        "workload": {
            "n_windows": n_windows,
            "n_samples": n_samples,
            "decimation": config.decimation,
            "adc_width": config.adc_width,
            "seed": seed,
        },
        "meta": {
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "toolchains": {
                "rustc": _tool_version(["rustc", "--version"]),
                "julia": _tool_version(["julia", "--version"]),
                "go": _tool_version(["go", "version"]),
                "mojo": _tool_version(["mojo", "--version"]),
            },
        },
        "backends": backends,
        "parity": parity,
        "host_context": host_context,
    }


def _print_table(report: dict[str, Any]) -> None:
    """Render the per-backend console summary."""
    header = f"{'Backend':<10}{'Samples/s':>18}{'Call (ms)':>14}{'Speedup':>10}{'Parity':>14}"
    print(header)
    print("-" * len(header))
    parity = report["parity"]
    for name in ("python", "rust", "mojo", "julia", "go"):
        info = report["backends"].get(name, {})
        if not info.get("used", False):
            print(f"{name:<10}{'MISSING':>18}{'':>14}{'':>10}   ({info.get('reason', '—')})")
            continue
        speed = info.get("speedup_over_python", 1.0)
        par = parity.get(name, {})
        par_text = (
            "reference"
            if name == parity["reference"]
            else ("bit-exact" if par.get("bit_exact") else f"Δ={par.get('max_abs_delta')}")
        )
        print(
            f"{name:<10}{info['samples_per_s']:>18,.0f}{info['median_call_ms']:>14.4f}"
            f"{speed:>9.2f}×{par_text:>14}"
        )


def main() -> int:
    """Parse arguments, run the benchmark and persist the JSON artefact."""
    parser = argparse.ArgumentParser(description="Benchmark the ADC-to-spike encoder.")
    parser.add_argument("--windows", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--python-repeats", type=int, default=3)
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "bench_adc_to_spike_kernel.json",
    )
    args = parser.parse_args()
    report = run(args.windows, args.seed, args.repeats, args.python_repeats)
    _print_table(report)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\nResults → {args.json.relative_to(REPO_ROOT)}")
    bit_exact = all(
        entry.get("bit_exact", True)
        for key, entry in report["parity"].items()
        if isinstance(entry, dict)
    )
    return 0 if bit_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
