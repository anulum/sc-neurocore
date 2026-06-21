#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benchmark for the dimensionality polyglot chain

"""Measure and compare the dimensionality backends.

``spike_train_pca``, ``demixed_pca`` (Kobak et al. 2016) and ``factor_analysis``
(Rubin & Thayer 1982) share a deterministic, sign-canonicalised covariance
eigendecomposition, so every backend agrees with the NumPy reference to
floating-point round-off. The benchmark times the three estimators on a fixed
spike-train population for each available backend, asserts parity against NumPy,
and records host-load and CPU affinity context per the benchmark-core-isolation
policy.

    taskset -c 10-11 python benchmarks/bench_dimensionality.py \\
        --json benchmarks/results/bench_dimensionality.json
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

_DIM = importlib.import_module("sc_neurocore.analysis.spike_stats.dimensionality")

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
        "rust": _DIM._rust_pca is not None,
        "julia": _DIM._ensure_julia_dim(),
        "go": _DIM._ensure_go_dim(),
        "mojo": _DIM._ensure_mojo_dim(),
    }


def _workload(
    n_neurons: int, n_steps: int, n_conditions: int, seed: int
) -> tuple[list[np.ndarray[Any, Any]], dict[int, list[np.ndarray[Any, Any]]]]:
    """A spike-train population (varied rates) plus a condition grouping."""
    rng = np.random.RandomState(seed)
    trains = [
        (rng.rand(n_steps) < (0.08 + 0.02 * (i % 7))).astype(np.int8) for i in range(n_neurons)
    ]
    per = max(1, n_neurons // n_conditions)
    conditions = {c: trains[c * per : (c + 1) * per] for c in range(n_conditions)}
    return trains, conditions


def _assess(
    trains: list[np.ndarray[Any, Any]],
    conditions: dict[int, list[np.ndarray[Any, Any]]],
    backend: str,
) -> dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    """Run the three estimators for one backend (the shared-eigendecomp workload)."""
    return {
        "pca": _DIM.spike_train_pca(trains, 3, 10, backend=backend),
        "demixed": _DIM.demixed_pca(conditions, 3, 10, backend=backend),
        "fa": _DIM.factor_analysis(trains, 3, 10, 30, backend=backend),
    }


def _time_backend(
    trains: list[np.ndarray[Any, Any]],
    conditions: dict[int, list[np.ndarray[Any, Any]]],
    backend: str,
    repeats: int,
) -> tuple[float, dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]]:
    """Return the median wall time (s) and the estimator outputs for one backend."""
    out = _assess(trains, conditions, backend)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = _assess(trains, conditions, backend)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2], out


def _max_diff(
    a: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
    b: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
) -> float:
    worst = 0.0
    for key in a:
        for ai, bi in zip(a[key], b[key]):
            if ai.size:
                worst = max(worst, float(np.max(np.abs(ai - bi))))
    return worst


def run(
    n_neurons: int, n_steps: int, n_conditions: int, seed: int, repeats: int
) -> dict[str, Any]:
    """Run every available dimensionality backend and return the report."""
    trains, conditions = _workload(n_neurons, n_steps, n_conditions, seed)
    availability = _availability()
    load_before = list(os.getloadavg())

    backends: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]] = {}
    for name in _BACKEND_ORDER:
        if not availability.get(name, False):
            backends[name] = {"available": False, "used": False, "reason": "backend not present"}
            continue
        runs = 5 if name == "julia" else repeats
        wall, out = _time_backend(trains, conditions, name, runs)
        outputs[name] = out
        backends[name] = {
            "available": True,
            "used": True,
            "median_call_ms": round(wall * 1e3, 6),
            "repeats": runs,
        }

    ref_ms = backends["python"]["median_call_ms"]
    for name in _BACKEND_ORDER:
        info = backends[name]
        if info.get("used") and name != "python":
            info["speedup_over_python"] = round(ref_ms / info["median_call_ms"], 3)

    parity: dict[str, Any] = {"reference": "python"}
    for name in _BACKEND_ORDER:
        if name == "python" or name not in outputs:
            continue
        parity[name] = {"max_abs_diff": _max_diff(outputs["python"], outputs[name])}

    load_after = list(os.getloadavg())
    return {
        "benchmark": "dimensionality_polyglot",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language_surfaces": ["Python", "Rust", "Julia", "Go", "Mojo"],
        "estimators": ["spike_train_pca", "demixed_pca", "factor_analysis"],
        "eigensolver": "LAPACK (NumPy/Rust/Julia), cyclic Jacobi (Go/Mojo)",
        "hardware_measurement_claimed": False,
        "workload": {
            "n_neurons": n_neurons,
            "n_steps": n_steps,
            "n_conditions": n_conditions,
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
        stats["max_abs_diff"] <= tol for name, stats in parity.items() if name != "reference"
    )


def main() -> int:
    """Parse arguments, run the benchmark and persist the JSON artefact."""
    parser = argparse.ArgumentParser(description="Benchmark the dimensionality polyglot chain.")
    parser.add_argument("--neurons", type=int, default=24)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--conditions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--parity-tol", type=float, default=1e-6)
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "bench_dimensionality.json",
    )
    args = parser.parse_args()
    report = run(args.neurons, args.steps, args.conditions, args.seed, args.repeats)

    print(f"{'Backend':<8}{'Call (ms)':>14}{'Speedup':>11}")
    print("-" * 33)
    for name in _BACKEND_ORDER:
        info = report["backends"].get(name, {})
        if not info.get("used"):
            print(f"{name:<8}{'MISSING':>14}")
            continue
        speed = info.get("speedup_over_python", 1.0)
        print(f"{name:<8}{info['median_call_ms']:>14.4f}{speed:>10.3f}×")

    print("\nParity vs NumPy reference (max abs diff over all estimators):")
    for name in _BACKEND_ORDER:
        stats = report["parity"].get(name)
        if stats is not None:
            print(f"  {name:<6} {stats['max_abs_diff']:.2e}")

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
