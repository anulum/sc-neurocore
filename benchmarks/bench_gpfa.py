#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benchmark for the GPFA EM polyglot chain (Python/Rust/Julia/Go/Mojo)

"""Measure and compare the GPFA EM backends from a shared deterministic init.

The PCA initialisation (:func:`gpfa_pca_init`) is computed once in Python and
handed to every backend, so the only differences between the NumPy reference and
the Rust, Julia, Go and Mojo paths are floating-point round-off in the dense
linear algebra. The benchmark times each available backend on a fixed workload,
asserts that every accelerated path agrees with NumPy within tolerance, and
records per-backend wall time plus host-load and CPU affinity context per the
benchmark-core-isolation policy.

    taskset -c 10-11 python benchmarks/bench_gpfa.py \\
        --json benchmarks/results/bench_gpfa.json
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
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "bridge"):
    _p = str(REPO_ROOT / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

_GPFA = importlib.import_module("sc_neurocore.analysis.spike_stats.gpfa")

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
    """Probe which GPFA backends can run in this environment."""
    return {
        "python": True,
        "rust": _GPFA._rust_gpfa_em is not None,
        "julia": _GPFA._ensure_julia_gpfa(),
        "go": _GPFA._ensure_go_gpfa(),
        "mojo": _GPFA._ensure_mojo_gpfa(),
    }


def _synthetic_trains(n_neurons: int, n_samples: int, seed: int) -> list[npt.NDArray[np.float64]]:
    """Deterministic parallel spike trains with neuron-specific slow modulation."""
    rng = np.random.default_rng(seed)
    trains: list[npt.NDArray[np.float64]] = []
    for i in range(n_neurons):
        rate = 0.05 * (1.0 + 0.5 * np.sin(np.arange(n_samples) / 30.0 + i))
        trains.append((rng.random(n_samples) < rate).astype(np.float64))
    return trains


def _time_backend(
    Y: npt.NDArray[np.float64],
    init: tuple[Any, Any, Any, Any],
    max_iter: int,
    tol: float,
    backend: str,
    repeats: int,
) -> tuple[float, tuple[Any, ...]]:
    """Return the median dispatch wall time (s) and the final EM result."""
    C0, d0, R0, tau = init
    result = _GPFA._gpfa_em_dispatch(Y, C0, d0, R0, tau, max_iter, tol, backend)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = _GPFA._gpfa_em_dispatch(Y, C0, d0, R0, tau, max_iter, tol, backend)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2], result


def run(
    n_neurons: int,
    n_samples: int,
    n_latents: int,
    bin_ms: float,
    max_iter: int,
    tol: float,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    """Run every available GPFA backend and return the benchmark report."""
    trains = _synthetic_trains(n_neurons, n_samples, seed)
    Y = _GPFA._bin_trains(trains, bin_ms, dt=0.001)
    n_bins = int(Y.shape[1])
    n_latents = min(n_latents, n_neurons, n_bins)
    init = _GPFA.gpfa_pca_init(Y, n_latents, bin_ms)

    availability = _availability()
    load_before = list(os.getloadavg())

    backends: dict[str, dict[str, Any]] = {}
    results: dict[str, tuple[Any, ...]] = {}
    for name in _BACKEND_ORDER:
        if not availability.get(name, False):
            backends[name] = {"available": False, "used": False, "reason": "backend not present"}
            continue
        runs = 3 if name == "python" else repeats
        wall, result = _time_backend(Y, init, max_iter, tol, name, runs)
        results[name] = result
        backends[name] = {
            "available": True,
            "used": True,
            "median_call_ms": round(wall * 1e3, 6),
            "em_iterations": len(result[4]),
            "repeats": runs,
        }

    ref_ms = backends["python"]["median_call_ms"]
    for name in _BACKEND_ORDER:
        info = backends[name]
        if info.get("used") and name != "python":
            info["speedup_over_python"] = round(ref_ms / info["median_call_ms"], 3)

    # Parity: every accelerated backend versus the NumPy reference.
    x_py, c_py, d_py, r_py, ll_py = results["python"]
    parity: dict[str, Any] = {"reference": "python"}
    for name in _BACKEND_ORDER:
        if name == "python" or name not in results:
            continue
        x_b, c_b, d_b, r_b, ll_b = results[name]
        parity[name] = {
            "trajectories_max_abs_diff": float(np.abs(x_b - x_py).max()),
            "C_max_abs_diff": float(np.abs(c_b - c_py).max()),
            "d_max_abs_diff": float(np.abs(d_b - d_py).max()),
            "R_max_abs_diff": float(np.abs(r_b - r_py).max()),
            "log_likelihood_max_abs_diff": float(np.abs(np.array(ll_b) - np.array(ll_py)).max()),
            "iterations_match": len(ll_b) == len(ll_py),
        }

    load_after = list(os.getloadavg())
    return {
        "benchmark": "gpfa_em_polyglot",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language_surfaces": ["Python", "Rust", "Julia", "Go", "Mojo"],
        "kernel": "gpfa_em_from_init",
        "hardware_measurement_claimed": False,
        "workload": {
            "n_neurons": n_neurons,
            "n_samples": n_samples,
            "n_bins": n_bins,
            "n_latents": n_latents,
            "bin_ms": bin_ms,
            "max_iter": max_iter,
            "tol": tol,
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
    ok = True
    for name, stats in parity.items():
        if name == "reference":
            continue
        worst = max(
            stats["trajectories_max_abs_diff"],
            stats["C_max_abs_diff"],
            stats["d_max_abs_diff"],
            stats["R_max_abs_diff"],
        )
        if worst > tol or not stats["iterations_match"]:
            ok = False
    return ok


def main() -> int:
    """Parse arguments, run the benchmark and persist the JSON artefact."""
    parser = argparse.ArgumentParser(description="Benchmark the GPFA EM polyglot chain.")
    parser.add_argument("--neurons", type=int, default=8)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--latents", type=int, default=3)
    parser.add_argument("--bin-ms", type=float, default=20.0)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--parity-tol", type=float, default=1e-6)
    parser.add_argument(
        "--json", type=Path, default=REPO_ROOT / "benchmarks" / "results" / "bench_gpfa.json"
    )
    args = parser.parse_args()
    report = run(
        args.neurons,
        args.samples,
        args.latents,
        args.bin_ms,
        args.max_iter,
        args.tol,
        args.seed,
        args.repeats,
    )

    print(f"{'Backend':<8}{'Call (ms)':>14}{'Iters':>8}{'Speedup':>11}")
    print("-" * 41)
    for name in _BACKEND_ORDER:
        info = report["backends"].get(name, {})
        if not info.get("used"):
            print(f"{name:<8}{'MISSING':>14}")
            continue
        speed = info.get("speedup_over_python", 1.0)
        print(f"{name:<8}{info['median_call_ms']:>14.4f}{info['em_iterations']:>8}{speed:>10.3f}×")

    print("\nParity vs NumPy reference (max abs diff):")
    for name in _BACKEND_ORDER:
        stats = report["parity"].get(name)
        if stats is None:
            continue
        print(
            f"  {name:<6} traj={stats['trajectories_max_abs_diff']:.2e} "
            f"C={stats['C_max_abs_diff']:.2e} "
            f"ll={stats['log_likelihood_max_abs_diff']:.2e}"
        )

    within = _parity_within_tolerance(report["parity"], args.parity_tol)
    print(f"\nAll accelerated backends within {args.parity_tol:.0e} of NumPy: {within}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Results → {args.json.relative_to(REPO_ROOT)}")
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())
