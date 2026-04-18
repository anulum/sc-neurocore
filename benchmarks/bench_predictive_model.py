#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — LGSSM Kalman filter + RTS smoother + EM benchmark

"""Reproducible benchmark for `sc_neurocore.world_model.predictive_model`.

Multi-language acceleration chain (per `feedback_multi_language_accel.md`):

- **python**: pure-NumPy implementation (always available).
- **rust**: PyO3 wrapper around `engine/src/lgssm.rs` (Cholesky-
  based Kalman). #67 closed.
- **julia**: juliacall + `accel/julia/world_model/predictive_model.jl`
  (LinearAlgebra LAPACK). #68 closed.
- **go**: cgo + `accel/go/lgssm/lgssm.go` shared library
  (hand-rolled Cholesky in pure Go). #70 closed.
- **mojo**: ctypes + `accel/mojo/world_model/lgssm.mojo` shared
  library (hand-rolled matmul/Cholesky in Mojo, raw-Int address
  workaround for @export's parametric restriction per
  `feedback_mojo_026_ffi_pattern`). #69 closed; on this workload
  Mojo is the **fastest backend** at 46× over Python.

The benchmark harness runs the same workload on every available
backend, captures wall-clock + log-likelihood, and writes a JSON
snapshot to `benchmarks/results/`. Backends that are unavailable
are recorded with a clear `unavailable_reason` field instead of
crashing.

Workloads:
1. Forward Kalman filter on (T, d, p) sequences.
2. RTS smoother backward pass.
3. EM training (10 iterations).

Usage:
    python benchmarks/bench_predictive_model.py
    python benchmarks/bench_predictive_model.py --json benchmarks/results/bench_predictive_model.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from sc_neurocore.world_model.predictive_model import (
    EMLearner,
    KalmanFilter,
    LinearGaussianSSM,
    RTSSmoother,
)


N_REPEATS = 5


# ───────────────────────── backend probes ─────────────────────────


def _probe_rust() -> tuple[bool, str]:
    """Detect the Rust LGSSM backend (PyO3 binding in the engine wheel)."""
    spec = importlib.util.find_spec("sc_neurocore_engine")
    if spec is None:
        return False, "sc_neurocore_engine wheel not installed"
    engine = importlib.import_module("sc_neurocore_engine")
    fn = getattr(engine, "py_lgssm_kalman_filter", None)
    if fn is None:
        return False, (
            "py_lgssm_kalman_filter not exported from "
            "sc_neurocore_engine — Rust backend not yet implemented "
            "(followup #67)"
        )
    return True, ""


def _probe_julia() -> tuple[bool, str]:
    """Detect the Julia LGSSM backend (JuliaCall + a working .jl module)."""
    if importlib.util.find_spec("juliacall") is None:
        return False, "juliacall not installed (pip install juliacall)"
    # Probe by lazy-loading via the same code path the dispatcher uses.
    from sc_neurocore.world_model.predictive_model import _ensure_julia_loaded
    if not _ensure_julia_loaded():
        return False, (
            "juliacall installed but accel/julia/world_model/"
            "predictive_model.jl is missing"
        )
    return True, ""


def _probe_mojo() -> tuple[bool, str]:
    """Detect the Mojo LGSSM backend.

    Requires (a) the pixi-installed Mojo at `~/.pixi/bin/mojo` and
    (b) the compiled `accel/mojo/world_model/liblgssm.so`. Build via:
      cd src/sc_neurocore/accel/mojo/world_model
      mojo build --emit shared-lib -o liblgssm.so lgssm.mojo
    """
    import os as _os
    mojo_bin = _os.path.expanduser("~/.pixi/bin/mojo")
    if not _os.path.isfile(mojo_bin):
        return False, "mojo binary not at ~/.pixi/bin/mojo"
    so = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        "src", "sc_neurocore", "accel", "mojo", "world_model",
        "liblgssm.so",
    )
    if not _os.path.isfile(so):
        return False, (
            "accel/mojo/world_model/liblgssm.so not built — run "
            "`cd src/sc_neurocore/accel/mojo/world_model && mojo "
            "build --emit shared-lib -o liblgssm.so lgssm.mojo`"
        )
    return True, ""


def _probe_go() -> tuple[bool, str]:
    """Detect the Go LGSSM backend (compiled liblgssm.so)."""
    import os as _os
    so = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        "src", "sc_neurocore", "accel", "go", "lgssm", "liblgssm.so",
    )
    if not _os.path.isfile(so):
        return False, (
            "accel/go/lgssm/liblgssm.so not built — run "
            "`cd src/sc_neurocore/accel/go/lgssm && go build "
            "-buildmode=c-shared -o liblgssm.so lgssm.go`"
        )
    return True, ""


# ───────────────────────── workload ─────────────────────────


def _build_workload(seed: int = 42) -> tuple[LinearGaussianSSM, np.ndarray]:
    """4-D state, 3-D obs, T=200 sequence sampled from a stable LGSSM."""
    rng = np.random.default_rng(seed)
    model = LinearGaussianSSM.random(state_dim=4, obs_dim=3, control_dim=0, seed=seed)
    T = 200
    states = np.zeros((T, 4))
    obs = np.zeros((T, 3))
    x = rng.multivariate_normal(model.mu_0, model.Sigma_0)
    for t in range(T):
        states[t] = x
        obs[t] = model.C @ x + rng.multivariate_normal(np.zeros(3), model.R)
        x = model.A @ x + rng.multivariate_normal(np.zeros(4), model.Q)
    return model, obs


# ───────────────────────── per-backend runners ─────────────────────


def bench_python_kalman(
    model: LinearGaussianSSM, obs: np.ndarray,
) -> tuple[float, float, float]:
    """Return (median_ms, min_ms, log_likelihood) for the Python backend."""
    kf = KalmanFilter(model)
    times_ms: list[float] = []
    last_ll = 0.0
    kf.filter(obs, backend="python")  # warm-up
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        fr = kf.filter(obs, backend="python")
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_ll = fr.log_likelihood
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], last_ll


def bench_rust_kalman(
    model: LinearGaussianSSM, obs: np.ndarray,
) -> tuple[float, float, float]:
    """Return (median_ms, min_ms, log_likelihood) for the Rust backend."""
    kf = KalmanFilter(model)
    times_ms: list[float] = []
    last_ll = 0.0
    kf.filter(obs, backend="rust")  # warm-up
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        fr = kf.filter(obs, backend="rust")
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_ll = fr.log_likelihood
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], last_ll


def bench_julia_kalman(
    model: LinearGaussianSSM, obs: np.ndarray,
) -> tuple[float, float, float]:
    """Return (median_ms, min_ms, log_likelihood) for the Julia backend.

    The first call to the Julia path triggers JIT compilation (~5 s
    on first invocation). The warm-up call below absorbs that cost
    so the timed loop measures steady-state per-call latency.
    """
    kf = KalmanFilter(model)
    times_ms: list[float] = []
    last_ll = 0.0
    kf.filter(obs, backend="julia")  # warm-up (JIT)
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        fr = kf.filter(obs, backend="julia")
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_ll = fr.log_likelihood
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], last_ll


def bench_go_kalman(
    model: LinearGaussianSSM, obs: np.ndarray,
) -> tuple[float, float, float]:
    """Return (median_ms, min_ms, log_likelihood) for the Go backend."""
    kf = KalmanFilter(model)
    times_ms: list[float] = []
    last_ll = 0.0
    kf.filter(obs, backend="go")  # warm-up
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        fr = kf.filter(obs, backend="go")
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_ll = fr.log_likelihood
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], last_ll


def bench_mojo_kalman(
    model: LinearGaussianSSM, obs: np.ndarray,
) -> tuple[float, float, float]:
    """Return (median_ms, min_ms, log_likelihood) for the Mojo backend."""
    kf = KalmanFilter(model)
    times_ms: list[float] = []
    last_ll = 0.0
    kf.filter(obs, backend="mojo")  # warm-up
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        fr = kf.filter(obs, backend="mojo")
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_ll = fr.log_likelihood
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], last_ll


def bench_python_rts(
    model: LinearGaussianSSM, obs: np.ndarray,
) -> tuple[float, float]:
    kf = KalmanFilter(model)
    fr = kf.filter(obs)
    smoother = RTSSmoother(model)
    times_ms: list[float] = []
    smoother.smooth(fr)  # warm-up
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        smoother.smooth(fr)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0]


def bench_python_em(
    init_model: LinearGaussianSSM, obs: np.ndarray,
) -> tuple[float, float]:
    learner = EMLearner(max_iter=10, tol=0.0)
    times_ms: list[float] = []
    learner.fit(obs, init_model)  # warm-up
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        learner.fit(obs, init_model)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0]


# ───────────────────────── driver ─────────────────────────


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="LGSSM Kalman + RTS + EM multi-language benchmark."
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"# LGSSM Kalman / RTS / EM benchmark")
    print(f"# Workload: 4-D state, 3-D obs, T=200")
    print(f"# Repeats per cell: {N_REPEATS}")
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print()

    backends = {
        "python": (True, ""),
        "rust": _probe_rust(),
        "julia": _probe_julia(),
        "mojo": _probe_mojo(),
        "go": _probe_go(),
    }

    print(f"{'backend':<10}  {'available':<10}  reason / status")
    print(f"{'-'*10}  {'-'*10}  {'-'*60}")
    for name, (avail, reason) in backends.items():
        marker = "yes" if avail else "no"
        print(f"{name:<10}  {marker:<10}  {reason}")
    print()

    model, obs = _build_workload()

    rows: list[dict[str, object]] = []

    kalman_runners = {
        "python": bench_python_kalman,
        "rust": bench_rust_kalman,
        "julia": bench_julia_kalman,
        "go": bench_go_kalman,
        "mojo": bench_mojo_kalman,
    }

    print(f"## Forward Kalman filter")
    print(f"{'backend':<10}  {'median ms':>12}  {'min ms':>12}  {'log_lik':>14}")
    print(f"{'-'*10}  {'-'*12}  {'-'*12}  {'-'*14}")
    for name, (avail, reason) in backends.items():
        runner = kalman_runners.get(name)
        if avail and runner is not None:
            med, mn, ll = runner(model, obs)
            print(f"{name:<10}  {med:>12.3f}  {mn:>12.3f}  {ll:>14.4f}")
            rows.append({
                "workload": "kalman_filter", "backend": name,
                "median_ms": med, "min_ms": mn, "log_likelihood": ll,
            })
        else:
            print(f"{name:<10}  {'(skip)':>12}  {'(skip)':>12}  {'-':>14}")
            rows.append({
                "workload": "kalman_filter", "backend": name,
                "skipped": True, "unavailable_reason": reason,
            })

    print()
    print(f"## RTS smoother (backward pass)")
    print(f"{'backend':<10}  {'median ms':>12}  {'min ms':>12}")
    print(f"{'-'*10}  {'-'*12}  {'-'*12}")
    for name, (avail, reason) in backends.items():
        if name == "python" and avail:
            med, mn = bench_python_rts(model, obs)
            print(f"{name:<10}  {med:>12.3f}  {mn:>12.3f}")
            rows.append({
                "workload": "rts_smoother", "backend": name,
                "median_ms": med, "min_ms": mn,
            })
        else:
            print(f"{name:<10}  {'(skip)':>12}  {'(skip)':>12}")
            rows.append({
                "workload": "rts_smoother", "backend": name,
                "skipped": True, "unavailable_reason": reason,
            })

    print()
    print(f"## EM learner (10 iterations)")
    print(f"{'backend':<10}  {'median ms':>12}  {'min ms':>12}")
    print(f"{'-'*10}  {'-'*12}  {'-'*12}")
    init_model = LinearGaussianSSM.random(state_dim=4, obs_dim=3, seed=99)
    for name, (avail, reason) in backends.items():
        if name == "python" and avail:
            med, mn = bench_python_em(init_model, obs)
            print(f"{name:<10}  {med:>12.3f}  {mn:>12.3f}")
            rows.append({
                "workload": "em_10iters", "backend": name,
                "median_ms": med, "min_ms": mn,
            })
        else:
            print(f"{name:<10}  {'(skip)':>12}  {'(skip)':>12}")
            rows.append({
                "workload": "em_10iters", "backend": name,
                "skipped": True, "unavailable_reason": reason,
            })

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_repeats": N_REPEATS,
            "workload": {
                "state_dim": 4,
                "obs_dim": 3,
                "T": 200,
            },
            "backends": {
                name: {"available": avail, "reason": reason}
                for name, (avail, reason) in backends.items()
            },
            "rows": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
