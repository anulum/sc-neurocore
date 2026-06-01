# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-backend benchmark for the Wong-Wang 2006 decision unit

"""Measure and compare Python / Rust / Julia / Go / Mojo batch simulators.

Writes a table to stdout and a JSON file under ``benchmarks/results/``
recording every backend's `available` / `used` / `reason` so downstream
consumers can see which backends participated and why any skipped.

Run:

    python benchmarks/bench_wong_wang.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sc_neurocore.neurons.models.wong_wang import WongWangUnit

BackendFn = Callable[
    [
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    dict[str, Any],
]

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PARAMS: dict[str, float] = dict(
    tau_s=0.1,
    gamma=0.641,
    j_n=0.2609,
    j_cross=0.0497,
    i_0=0.3255,
    sigma=0.02,
    dt=0.001,
)
INTEGRATOR = "fixed_step_rk4_piecewise_constant_noise"

N_STEPS = 100_000
SEED = 42
PARITY_N = 5_000


# ── Backend probes ────────────────────────────────────────────────────


def _probe_rust() -> tuple[BackendFn | None, str]:
    try:
        from sc_neurocore_engine import py_wong_wang_simulate

        return py_wong_wang_simulate, "available"
    except ImportError as e:
        return None, f"missing: {e}"


def _probe_julia() -> tuple[BackendFn | None, str]:
    try:
        from sc_neurocore.accel.julia.neurons import (
            _HAS_JULIA_NEURONS,
            simulate_wong_wang,
        )

        if not _HAS_JULIA_NEURONS:
            return None, "juliacall import failed"
        return simulate_wong_wang, "available"
    except ImportError as e:
        return None, f"missing: {e}"


def _probe_go() -> tuple[BackendFn | None, str]:
    try:
        from sc_neurocore.accel.go.wong_wang import (
            _HAS_GO_WONG_WANG,
            simulate_wong_wang,
        )

        if not _HAS_GO_WONG_WANG:
            return (
                None,
                "libwong_wang.so not built — run go build in accel/go/wong_wang",
            )
        return simulate_wong_wang, "available"
    except ImportError as e:
        return None, f"missing: {e}"


def _probe_mojo() -> tuple[BackendFn | None, str]:
    try:
        from sc_neurocore.accel.mojo.wong_wang import (
            _HAS_MOJO_WONG_WANG,
            simulate_wong_wang,
        )

        if not _HAS_MOJO_WONG_WANG:
            return (
                None,
                "libwong_wang.so not built — run mojo build in accel/mojo/wong_wang",
            )
        return simulate_wong_wang, "available"
    except ImportError as e:
        return None, f"missing: {e}"


# ── Workload runners ──────────────────────────────────────────────────


def _run_python(n: int, stim1: np.ndarray, stim2: np.ndarray, seed: int) -> float:
    np.random.seed(seed)
    u = WongWangUnit(**DEFAULT_PARAMS)
    t0 = time.perf_counter()
    for t in range(n):
        u.step(float(stim1[t]), float(stim2[t]))
    return time.perf_counter() - t0


def _run_batch(fn: BackendFn, n: int, stim1: np.ndarray, stim2: np.ndarray, seed: int) -> tuple[float, dict[str, Any]]:
    np.random.seed(seed)
    xi = np.random.randn(2 * n).astype(np.float64)
    # Warm-up (JIT / first-call dispatch overhead)
    fn(
        0.1,
        0.1,
        DEFAULT_PARAMS["tau_s"],
        DEFAULT_PARAMS["gamma"],
        DEFAULT_PARAMS["j_n"],
        DEFAULT_PARAMS["j_cross"],
        DEFAULT_PARAMS["i_0"],
        DEFAULT_PARAMS["sigma"],
        DEFAULT_PARAMS["dt"],
        stim1[:1_000],
        stim2[:1_000],
        xi[: 2 * 1_000],
    )
    # Timed run
    np.random.seed(seed)
    xi = np.random.randn(2 * n).astype(np.float64)
    t0 = time.perf_counter()
    out = fn(
        0.1,
        0.1,
        DEFAULT_PARAMS["tau_s"],
        DEFAULT_PARAMS["gamma"],
        DEFAULT_PARAMS["j_n"],
        DEFAULT_PARAMS["j_cross"],
        DEFAULT_PARAMS["i_0"],
        DEFAULT_PARAMS["sigma"],
        DEFAULT_PARAMS["dt"],
        stim1,
        stim2,
        xi,
    )
    return time.perf_counter() - t0, out


# ── Parity check ──────────────────────────────────────────────────────


def _parity_trace(fn: BackendFn, n: int, stim1: np.ndarray, stim2: np.ndarray, seed: int) -> dict[str, Any]:
    np.random.seed(seed)
    xi = np.random.randn(2 * n).astype(np.float64)
    return fn(
        0.1,
        0.1,
        DEFAULT_PARAMS["tau_s"],
        DEFAULT_PARAMS["gamma"],
        DEFAULT_PARAMS["j_n"],
        DEFAULT_PARAMS["j_cross"],
        DEFAULT_PARAMS["i_0"],
        DEFAULT_PARAMS["sigma"],
        DEFAULT_PARAMS["dt"],
        stim1,
        stim2,
        xi,
    )


# ── Main ──────────────────────────────────────────────────────────────


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> int:
    stim1 = np.full(N_STEPS, 0.1, dtype=np.float64)
    stim2 = np.zeros(N_STEPS, dtype=np.float64)

    backends: dict[str, dict[str, Any]] = {}

    # Python primary — always available.
    t_py = _run_python(N_STEPS, stim1, stim2, SEED)
    backends["python"] = {
        "available": True,
        "used": True,
        "reason": "primary reference",
        "wall_ms": round(t_py * 1e3, 3),
        "steps_per_s": round(N_STEPS / t_py, 0),
    }

    # Probe + run each accelerator.
    probes: list[tuple[str, Callable[[], tuple[BackendFn | None, str]]]] = [
        ("rust", _probe_rust),
        ("julia", _probe_julia),
        ("go", _probe_go),
        ("mojo", _probe_mojo),
    ]
    outs: dict[str, Any] = {}
    for name, probe in probes:
        fn, reason = probe()
        if fn is None:
            backends[name] = {"available": False, "used": False, "reason": reason}
            continue
        wall, out = _run_batch(fn, N_STEPS, stim1, stim2, SEED)
        backends[name] = {
            "available": True,
            "used": True,
            "reason": reason,
            "wall_ms": round(wall * 1e3, 3),
            "steps_per_s": round(N_STEPS / wall, 0),
            "speedup_over_python": round(t_py / wall, 2),
        }
        outs[name] = out

    # Parity: every backend vs rust (if present) bit-exact on shared xi;
    # libm-vs-f64::exp drift tolerated for mojo.
    ref_name = "rust" if "rust" in outs else (next(iter(outs), None) if outs else None)
    parity: dict[str, Any] = {"reference": ref_name}
    if ref_name is not None:
        p_stim1 = np.full(PARITY_N, 0.1, dtype=np.float64)
        p_stim2 = np.zeros(PARITY_N, dtype=np.float64)
        ref_fn = dict(probes)[ref_name]()[0]
        assert ref_fn is not None
        ref_out = _parity_trace(ref_fn, PARITY_N, p_stim1, p_stim2, SEED)
        for name, probe in probes:
            if name == ref_name:
                parity[name] = {"atol": 0.0}
                continue
            fn = backends[name].get("available") and probe()[0]
            if not fn:
                parity[name] = {"atol": "skipped"}
                continue
            out = _parity_trace(fn, PARITY_N, p_stim1, p_stim2, SEED)
            d_s1 = float(np.abs(ref_out["s1"] - out["s1"]).max())
            d_s2 = float(np.abs(ref_out["s2"] - out["s2"]).max())
            parity[name] = {
                "max_abs_delta_s1": d_s1,
                "max_abs_delta_s2": d_s2,
                "atol": max(d_s1, d_s2),
            }

    # Print console table.
    header = f"{'Backend':<10} {'Steps/s':>15} {'Wall (ms)':>12} {'Speedup':>10} {'Parity':>14}"
    print(header)
    print("-" * len(header))
    for name in ("python", "rust", "julia", "go", "mojo"):
        b = backends.get(name, {})
        if not b.get("used", False):
            reason = b.get("reason", "—")
            print(f"{name:<10} {'MISSING':>15} {'':>12} {'':>10}   ({reason})")
            continue
        sp = b.get("speedup_over_python", 1.0)
        steps = b.get("steps_per_s", 0)
        wall = b.get("wall_ms", 0)
        par = parity.get(name, {})
        if name == parity.get("reference"):
            par_text = "reference"
        else:
            atol = par.get("atol", "—")
            par_text = f"Δ={atol:.1e}" if isinstance(atol, float) else str(atol)
        print(f"{name:<10} {int(steps):>15,} {wall:>12.2f} {sp:>9.2f}× {par_text:>14}")

    # Capture environment metadata.
    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": _cpu_model(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "n_steps": N_STEPS,
        "parity_n": PARITY_N,
        "seed": SEED,
        "params": DEFAULT_PARAMS,
        "integrator": INTEGRATOR,
    }

    out_path = RESULTS_DIR / "bench_wong_wang.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "meta": meta,
                "backends": backends,
                "parity": parity,
            },
            f,
            indent=2,
            sort_keys=False,
        )
    print(f"\nResults → {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
