#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-backend benchmark for the mixed-precision dense MAC

"""Measure and compare Python / Rust / Julia / Go / Mojo mixed-dense kernels.

The integer mixed-precision Q8.8 × Q16.16 dense MAC is exact, so every backend is
bit-identical to the Python floor (parity tolerance zero). The benchmark records,
per backend, ``available`` / ``used`` / ``reason`` plus per-call wall time and
throughput, and writes a JSON artefact under ``benchmarks/results/``.

Host-load and CPU-affinity context is captured per the benchmark-core-isolation
policy. Run pinned for production evidence::

    taskset -c 10-11 python benchmarks/bench_mixed_dense_kernel.py \\
        --json benchmarks/results/bench_mixed_dense_kernel.json

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

from sc_neurocore.compiler.mixed_dense_kernel import (  # noqa: E402
    MixedDenseBatchResult,
    available_backends,
    mixed_dense_forward_batch,
    mixed_dense_forward_batch_q88_q1616,
)

BackendFn = Callable[[npt.ArrayLike, npt.ArrayLike, int, int], MixedDenseBatchResult]

_RESULT_FIELDS = ("outputs_q1616", "overflow", "underflow")


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
    n_outputs: int, n_inputs: int, n_batch: int, seed: int
) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int32]]:
    """Build a deterministic mixed-dense workload within the int64 contraction bound."""
    rng = np.random.default_rng(seed)
    weights = rng.integers(-32768, 32768, size=n_outputs * n_inputs, dtype=np.int16)
    inputs = rng.integers(-(1 << 20), 1 << 20, size=n_batch * n_inputs, dtype=np.int32)
    return weights, inputs


def _time_backend(
    fn: BackendFn,
    weights: npt.NDArray[np.int16],
    inputs: npt.NDArray[np.int32],
    n_outputs: int,
    n_inputs: int,
    repeats: int,
) -> tuple[float, MixedDenseBatchResult]:
    """Warm up once then return the median per-call wall time and a result."""
    result = fn(weights, inputs, n_outputs, n_inputs)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(weights, inputs, n_outputs, n_inputs)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2], result


def _backend_fn(name: str) -> BackendFn:
    """Return the uniform call wrapper for a backend name."""
    if name == "python":
        return mixed_dense_forward_batch_q88_q1616

    def _call(
        weights: npt.ArrayLike, inputs: npt.ArrayLike, n_outputs: int, n_inputs: int
    ) -> MixedDenseBatchResult:
        return mixed_dense_forward_batch(weights, inputs, n_outputs, n_inputs, backend=name)

    return _call


def _parity_delta(reference: MixedDenseBatchResult, candidate: MixedDenseBatchResult) -> int:
    """Maximum absolute element difference across all result arrays (zero when bit-exact)."""
    delta = 0
    for field in _RESULT_FIELDS:
        ref = np.asarray(getattr(reference, field), dtype=np.int64)
        cand = np.asarray(getattr(candidate, field), dtype=np.int64)
        delta = max(delta, int(np.abs(ref - cand).max()))
    return delta


def run(
    n_outputs: int, n_inputs: int, n_batch: int, seed: int, repeats: int, python_repeats: int
) -> dict[str, Any]:
    """Run every available backend and return the benchmark report."""
    weights, inputs = _make_workload(n_outputs, n_inputs, n_batch, seed)
    macs = n_batch * n_outputs * n_inputs
    availability = available_backends()
    load_before = list(os.getloadavg())

    backends: dict[str, dict[str, Any]] = {}
    results: dict[str, MixedDenseBatchResult] = {}
    order = ("python", "rust", "mojo", "julia", "go")
    for name in order:
        if not availability.get(name, False):
            backends[name] = {"available": False, "used": False, "reason": "backend probe failed"}
            continue
        runs = python_repeats if name == "python" else repeats
        wall, result = _time_backend(_backend_fn(name), weights, inputs, n_outputs, n_inputs, runs)
        results[name] = result
        backends[name] = {
            "available": True,
            "used": True,
            "reason": "primary reference" if name == "python" else "available",
            "median_call_ms": round(wall * 1e3, 6),
            "mac_per_s": round(macs / wall, 1),
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
        "benchmark": "mixed_dense_kernel_batch",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language_surfaces": ["Python", "Rust", "Julia", "Go", "Mojo"],
        "kernel": "mixed_dense_forward_batch_q88_q1616",
        "hardware_measurement_claimed": False,
        "benchmark_isolation_mode": host_context["isolation_mode"],
        "workload": {
            "n_outputs": n_outputs,
            "n_inputs": n_inputs,
            "n_batch": n_batch,
            "multiply_accumulates": macs,
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
    header = f"{'Backend':<10}{'MAC/s':>18}{'Call (ms)':>14}{'Speedup':>10}{'Parity':>14}"
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
            f"{name:<10}{info['mac_per_s']:>18,.0f}{info['median_call_ms']:>14.4f}"
            f"{speed:>9.2f}×{par_text:>14}"
        )


def main() -> int:
    """Parse arguments, run the benchmark and persist the JSON artefact."""
    parser = argparse.ArgumentParser(description="Benchmark the mixed-precision dense MAC.")
    parser.add_argument("--outputs", type=int, default=256)
    parser.add_argument("--inputs", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--python-repeats", type=int, default=5)
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "bench_mixed_dense_kernel.json",
    )
    args = parser.parse_args()
    report = run(
        args.outputs, args.inputs, args.batch, args.seed, args.repeats, args.python_repeats
    )
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
