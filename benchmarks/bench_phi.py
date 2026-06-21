#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benchmark for the Phi* polyglot chain (Python/Rust/Julia/Go/Mojo)

"""Measure and compare the Phi* backends on a fixed workload.

Every backend computes the same Gaussian geometric estimator with the Cholesky
log-determinant form, so they agree up to floating-point round-off. The benchmark
times each available backend, asserts that every accelerated path matches the
NumPy reference within tolerance, and records per-backend wall time plus host-load
and CPU affinity context per the benchmark-core-isolation policy.

    taskset -c 10-11 python benchmarks/bench_phi.py \\
        --json benchmarks/results/bench_phi.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "bridge"):
    _p = str(REPO_ROOT / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PHI = importlib.import_module("sc_neurocore.analysis.phi_estimation")

_BACKEND_ORDER = ("python", "rust", "julia", "go", "mojo")


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


def _cpu_model() -> str:
    info = _read_text(Path("/proc/cpuinfo"))
    if info is not None:
        for line in info.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _host_context(load_before: list[float], load_after: list[float]) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    cpuset = _cpuset()
    shielded = cpuset == "10-11" or affinity == [10, 11]
    return {
        "affinity_cpus": affinity,
        "cgroup_effective_cpuset": cpuset,
        "load_average_before": load_before,
        "load_average_after": load_after,
        "runtime_cpuset_shield_claimed": shielded,
        "isolation_mode": "runtime-cpuset-shield" if shielded else "non-isolated-shared-host",
    }


def _availability() -> dict[str, bool]:
    return {
        "python": True,
        "rust": _PHI._rust_phi is not None,
        "julia": _PHI._ensure_julia_phi(),
        "go": _PHI._ensure_go_phi(),
        "mojo": _PHI._ensure_mojo_phi(),
    }


def _workload(n_channels: int, n_timesteps: int, seed: int) -> np.ndarray[Any, Any]:
    """Channels sharing a latent drive, so Phi* is non-trivial (positive)."""
    rng = np.random.RandomState(seed)
    shared = rng.randn(n_timesteps)
    chans = [shared * (0.4 + 0.1 * i) + 0.5 * rng.randn(n_timesteps) for i in range(n_channels)]
    return np.ascontiguousarray(np.vstack(chans), dtype=np.float64)


def _time_backend(
    data: np.ndarray[Any, Any], tau: int, backend: str, repeats: int
) -> tuple[float, float]:
    """Return the median wall time (s) and the Phi* value for one backend."""
    value = _PHI.phi_star(data, tau=tau, backend=backend)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = _PHI.phi_star(data, tau=tau, backend=backend)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2], value


def run(n_channels: int, n_timesteps: int, tau: int, seed: int, repeats: int) -> dict[str, Any]:
    """Run every available Phi* backend and return the benchmark report."""
    data = _workload(n_channels, n_timesteps, seed)
    availability = _availability()
    load_before = list(os.getloadavg())

    backends: dict[str, dict[str, Any]] = {}
    values: dict[str, float] = {}
    for name in _BACKEND_ORDER:
        if not availability.get(name, False):
            backends[name] = {"available": False, "used": False, "reason": "backend not present"}
            continue
        runs = 5 if name == "julia" else repeats
        wall, value = _time_backend(data, tau, name, runs)
        values[name] = value
        backends[name] = {
            "available": True,
            "used": True,
            "median_call_ms": round(wall * 1e3, 6),
            "phi": value,
            "repeats": runs,
        }

    ref_ms = backends["python"]["median_call_ms"]
    for name in _BACKEND_ORDER:
        info = backends[name]
        if info.get("used") and name != "python":
            info["speedup_over_python"] = round(ref_ms / info["median_call_ms"], 3)

    parity: dict[str, Any] = {"reference": "python"}
    for name in _BACKEND_ORDER:
        if name == "python" or name not in values:
            continue
        parity[name] = {"phi_abs_diff": float(abs(values[name] - values["python"]))}

    load_after = list(os.getloadavg())
    return {
        "benchmark": "phi_star_polyglot",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language_surfaces": ["Python", "Rust", "Julia", "Go", "Mojo"],
        "kernel": "phi_star",
        "hardware_measurement_claimed": False,
        "workload": {
            "n_channels": n_channels,
            "n_timesteps": n_timesteps,
            "tau": tau,
            "seed": seed,
        },
        "meta": {
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "backends": backends,
        "parity": parity,
        "host_context": _host_context(load_before, load_after),
    }


def _parity_within_tolerance(parity: dict[str, Any], tol: float) -> bool:
    """True when every accelerated backend agrees with NumPy within *tol*."""
    return all(
        stats["phi_abs_diff"] <= tol for name, stats in parity.items() if name != "reference"
    )


def main() -> int:
    """Parse arguments, run the benchmark and persist the JSON artefact."""
    parser = argparse.ArgumentParser(description="Benchmark the Phi* polyglot chain.")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--parity-tol", type=float, default=1e-6)
    parser.add_argument(
        "--json", type=Path, default=REPO_ROOT / "benchmarks" / "results" / "bench_phi.json"
    )
    args = parser.parse_args()
    report = run(args.channels, args.timesteps, args.tau, args.seed, args.repeats)

    print(f"{'Backend':<8}{'Call (ms)':>14}{'Phi':>12}{'Speedup':>11}")
    print("-" * 45)
    for name in _BACKEND_ORDER:
        info = report["backends"].get(name, {})
        if not info.get("used"):
            print(f"{name:<8}{'MISSING':>14}")
            continue
        speed = info.get("speedup_over_python", 1.0)
        print(f"{name:<8}{info['median_call_ms']:>14.4f}{info['phi']:>12.6f}{speed:>10.3f}×")

    print("\nParity vs NumPy reference (Phi abs diff):")
    for name in _BACKEND_ORDER:
        stats = report["parity"].get(name)
        if stats is not None:
            print(f"  {name:<6} {stats['phi_abs_diff']:.2e}")

    within = _parity_within_tolerance(report["parity"], args.parity_tol)
    print(f"\nAll accelerated backends within {args.parity_tol:.0e} of NumPy: {within}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Results → {args.json.relative_to(REPO_ROOT)}")
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())
