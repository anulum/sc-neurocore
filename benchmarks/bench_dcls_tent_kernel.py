#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-backend benchmark for the DCLS-max Q8.8 tent kernel

"""Measure and compare Python / Rust / Julia / Go / Mojo DCLS-max tent kernels.

The kernel is exact integer Q8.8 arithmetic, so every backend is bit-identical
to the Python floor (parity tolerance zero). The benchmark records, per backend,
``available`` / ``used`` / ``reason`` plus per-call wall time and throughput, and
writes a JSON artefact under ``benchmarks/results/``.

Host-load and CPU-affinity context is captured per the benchmark-core-isolation
policy. Run pinned for production evidence::

    taskset -c 10-11 python benchmarks/bench_dcls_tent_kernel.py \\
        --json benchmarks/results/bench_dcls_tent_kernel.json

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

from sc_neurocore.scpn.dcls_tent_kernel import (  # noqa: E402
    DclsBatchResult,
    available_backends,
    dcls_max_forward_batch,
    dcls_max_forward_batch_q88,
)

BackendFn = Callable[
    [npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, int], DclsBatchResult
]

_RESULT_FIELDS = (
    "outputs_q88",
    "accumulators_q16_16",
    "overflow",
    "active_tap_counts",
    "max_gates_q88",
)


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


def _make_workload(
    n_channels: int, n_taps: int, seed: int
) -> tuple[
    npt.NDArray[np.uint8], npt.NDArray[np.int16], npt.NDArray[np.int16], npt.NDArray[np.int16]
]:
    """Build a deterministic DCLS batch workload."""
    rng = np.random.default_rng(seed)
    total = n_channels * n_taps
    spikes = (rng.random(total) < 0.5).astype(np.uint8)
    weights = rng.integers(-32768, 32768, size=total, dtype=np.int16)
    centres = rng.integers(-256, (n_taps << 8) + 256, size=n_channels, dtype=np.int16)
    sigmas = rng.integers(1, (n_taps << 8) + 256, size=n_channels, dtype=np.int16)
    return spikes, weights, centres, sigmas


def _time_backend(
    fn: BackendFn,
    spikes: npt.NDArray[np.uint8],
    weights: npt.NDArray[np.int16],
    centres: npt.NDArray[np.int16],
    sigmas: npt.NDArray[np.int16],
    n_taps: int,
    repeats: int,
) -> tuple[float, DclsBatchResult]:
    """Warm up once then return the median per-call wall time and a result."""
    result = fn(spikes, weights, centres, sigmas, n_taps)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(spikes, weights, centres, sigmas, n_taps)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2], result


def _backend_fn(name: str) -> BackendFn:
    """Return the uniform call wrapper for a backend name."""
    if name == "python":
        return dcls_max_forward_batch_q88

    def _call(
        spikes: npt.ArrayLike,
        weights: npt.ArrayLike,
        centres: npt.ArrayLike,
        sigmas: npt.ArrayLike,
        n_taps: int,
    ) -> DclsBatchResult:
        return dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend=name)

    return _call


def _parity_delta(reference: DclsBatchResult, candidate: DclsBatchResult) -> int:
    """Maximum absolute element difference across all result arrays (zero when bit-exact)."""
    delta = 0
    for field in _RESULT_FIELDS:
        ref = np.asarray(getattr(reference, field), dtype=np.int64)
        cand = np.asarray(getattr(candidate, field), dtype=np.int64)
        delta = max(delta, int(np.abs(ref - cand).max()))
    return delta


def run(
    n_channels: int, n_taps: int, seed: int, repeats: int, python_repeats: int
) -> dict[str, Any]:
    """Run every available backend and return the benchmark report."""
    spikes, weights, centres, sigmas = _make_workload(n_channels, n_taps, seed)
    elements = n_channels * n_taps
    availability = available_backends()
    load_before = list(os.getloadavg())

    backends: dict[str, dict[str, Any]] = {}
    results: dict[str, DclsBatchResult] = {}
    order = ("python", "rust", "mojo", "julia", "go")
    for name in order:
        if not availability.get(name, False):
            backends[name] = {"available": False, "used": False, "reason": "backend probe failed"}
            continue
        runs = python_repeats if name == "python" else repeats
        wall, result = _time_backend(
            _backend_fn(name), spikes, weights, centres, sigmas, n_taps, runs
        )
        results[name] = result
        backends[name] = {
            "available": True,
            "used": True,
            "reason": "primary reference" if name == "python" else "available",
            "median_call_ms": round(wall * 1e3, 6),
            "channels_per_s": round(n_channels / wall, 1),
            "elements_per_s": round(elements / wall, 1),
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
        "benchmark": "dcls_tent_kernel_batch",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language_surfaces": ["Python", "Rust", "Julia", "Go", "Mojo"],
        "kernel": "dcls_max_forward_batch_q88",
        "hardware_measurement_claimed": False,
        "benchmark_isolation_mode": host_context["isolation_mode"],
        "workload": {
            "n_channels": n_channels,
            "n_taps": n_taps,
            "elements": elements,
            "seed": seed,
            "spike_density": 0.5,
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
    header = f"{'Backend':<10}{'Channels/s':>16}{'Call (ms)':>14}{'Speedup':>10}{'Parity':>14}"
    print(header)
    print("-" * len(header))
    parity = report["parity"]
    for name in ("python", "rust", "mojo", "julia", "go"):
        info = report["backends"].get(name, {})
        if not info.get("used", False):
            print(f"{name:<10}{'MISSING':>16}{'':>14}{'':>10}   ({info.get('reason', '—')})")
            continue
        speed = info.get("speedup_over_python", 1.0)
        par = parity.get(name, {})
        par_text = (
            "reference"
            if name == parity["reference"]
            else ("bit-exact" if par.get("bit_exact") else f"Δ={par.get('max_abs_delta')}")
        )
        print(
            f"{name:<10}{info['channels_per_s']:>16,.0f}{info['median_call_ms']:>14.4f}"
            f"{speed:>9.2f}×{par_text:>14}"
        )


def main() -> int:
    """Parse arguments, run the benchmark and persist the JSON artefact."""
    parser = argparse.ArgumentParser(description="Benchmark the DCLS-max Q8.8 tent kernel.")
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--taps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--python-repeats", type=int, default=3)
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "bench_dcls_tent_kernel.json",
    )
    args = parser.parse_args()
    report = run(args.channels, args.taps, args.seed, args.repeats, args.python_repeats)
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
