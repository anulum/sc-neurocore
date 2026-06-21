#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benchmark for the sorting-quality Mahalanobis polyglot chain

"""Measure and compare the Mahalanobis sorting-quality backends.

``isolation_distance`` (Harris et al. 2001) and ``l_ratio`` (Schmitzer-Torbert
et al. 2005) share one Cholesky-solve kernel for the squared Mahalanobis
distance, so every backend agrees with the NumPy reference up to floating-point
round-off. The benchmark times a combined cluster assessment (both metrics) on a
fixed workload for each available backend, asserts parity against NumPy, and
records host-load and CPU affinity context per the benchmark-core-isolation
policy.

    taskset -c 10-11 python benchmarks/bench_sorting_quality.py \\
        --json benchmarks/results/bench_sorting_quality.json
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

_SQ = importlib.import_module("sc_neurocore.analysis.spike_stats.sorting_quality")

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
        "rust": _SQ._rust_isolation is not None,
        "julia": _SQ._ensure_julia_sq(),
        "go": _SQ._ensure_go_sq(),
        "mojo": _SQ._ensure_mojo_sq(),
    }


def _workload(
    n_cluster: int, n_noise: int, n_features: int, seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """A compact cluster and an offset, broader noise cloud in feature space."""
    rng = np.random.RandomState(seed)
    cov = rng.randn(n_features, n_features)
    cov = cov @ cov.T / n_features + np.eye(n_features)
    chol = np.linalg.cholesky(cov)
    cluster = (rng.randn(n_cluster, n_features) @ chol.T).astype(np.float64)
    noise = (3.0 + 1.5 * rng.randn(n_noise, n_features)).astype(np.float64)
    return np.ascontiguousarray(cluster), np.ascontiguousarray(noise)


def _assess(cluster: np.ndarray[Any, Any], noise: np.ndarray[Any, Any], backend: str) -> tuple[float, float]:
    """Both Mahalanobis metrics for one backend (the shared-kernel workload)."""
    iso = _SQ.isolation_distance(cluster, noise, backend=backend)
    lr = _SQ.l_ratio(cluster, noise, backend=backend)
    return iso, lr


def _time_backend(
    cluster: np.ndarray[Any, Any], noise: np.ndarray[Any, Any], backend: str, repeats: int
) -> tuple[float, tuple[float, float]]:
    """Return the median wall time (s) and the (isolation, l_ratio) values."""
    values = _assess(cluster, noise, backend)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        values = _assess(cluster, noise, backend)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2], values


def run(
    n_cluster: int, n_noise: int, n_features: int, seed: int, repeats: int
) -> dict[str, Any]:
    """Run every available sorting-quality backend and return the report."""
    cluster, noise = _workload(n_cluster, n_noise, n_features, seed)
    availability = _availability()
    load_before = list(os.getloadavg())

    backends: dict[str, dict[str, Any]] = {}
    values: dict[str, tuple[float, float]] = {}
    for name in _BACKEND_ORDER:
        if not availability.get(name, False):
            backends[name] = {"available": False, "used": False, "reason": "backend not present"}
            continue
        runs = 5 if name == "julia" else repeats
        wall, (iso, lr) = _time_backend(cluster, noise, name, runs)
        values[name] = (iso, lr)
        backends[name] = {
            "available": True,
            "used": True,
            "median_call_ms": round(wall * 1e3, 6),
            "isolation_distance": iso,
            "l_ratio": lr,
            "repeats": runs,
        }

    ref_ms = backends["python"]["median_call_ms"]
    for name in _BACKEND_ORDER:
        info = backends[name]
        if info.get("used") and name != "python":
            info["speedup_over_python"] = round(ref_ms / info["median_call_ms"], 3)

    parity: dict[str, Any] = {"reference": "python"}
    ref_iso, ref_lr = values["python"]
    for name in _BACKEND_ORDER:
        if name == "python" or name not in values:
            continue
        iso, lr = values[name]
        parity[name] = {
            "isolation_distance_abs_diff": float(abs(iso - ref_iso)),
            "l_ratio_abs_diff": float(abs(lr - ref_lr)),
        }

    load_after = list(os.getloadavg())
    return {
        "benchmark": "sorting_quality_mahalanobis_polyglot",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language_surfaces": ["Python", "Rust", "Julia", "Go", "Mojo"],
        "kernel": "cluster_mahalanobis_sq (Cholesky solve)",
        "metrics": ["isolation_distance", "l_ratio"],
        "hardware_measurement_claimed": False,
        "workload": {
            "n_cluster": n_cluster,
            "n_noise": n_noise,
            "n_features": n_features,
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
        stats["isolation_distance_abs_diff"] <= tol and stats["l_ratio_abs_diff"] <= tol
        for name, stats in parity.items()
        if name != "reference"
    )


def main() -> int:
    """Parse arguments, run the benchmark and persist the JSON artefact."""
    parser = argparse.ArgumentParser(description="Benchmark the sorting-quality polyglot chain.")
    parser.add_argument("--cluster", type=int, default=64)
    parser.add_argument("--noise", type=int, default=256)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--parity-tol", type=float, default=1e-6)
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "bench_sorting_quality.json",
    )
    args = parser.parse_args()
    report = run(args.cluster, args.noise, args.features, args.seed, args.repeats)

    print(f"{'Backend':<8}{'Call (ms)':>14}{'IsoDist':>12}{'L-ratio':>12}{'Speedup':>11}")
    print("-" * 57)
    for name in _BACKEND_ORDER:
        info = report["backends"].get(name, {})
        if not info.get("used"):
            print(f"{name:<8}{'MISSING':>14}")
            continue
        speed = info.get("speedup_over_python", 1.0)
        print(
            f"{name:<8}{info['median_call_ms']:>14.4f}"
            f"{info['isolation_distance']:>12.4f}{info['l_ratio']:>12.6f}{speed:>10.3f}×"
        )

    print("\nParity vs NumPy reference (abs diff):")
    for name in _BACKEND_ORDER:
        stats = report["parity"].get(name)
        if stats is not None:
            print(
                f"  {name:<6} iso={stats['isolation_distance_abs_diff']:.2e}"
                f"  l_ratio={stats['l_ratio_abs_diff']:.2e}"
            )

    within = _parity_within_tolerance(report["parity"], args.parity_tol)
    print(f"\nAll accelerated backends within {args.parity_tol:.0e} of NumPy: {within}")

    out_path = args.json.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    try:
        shown: Path = out_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_path
    print(f"Results → {shown}")
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())
