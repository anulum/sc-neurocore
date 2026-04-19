# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-backend benchmark for the Wilson-Cowan 1972 E/I model

"""Measure + compare Python / Rust / Julia / Go / Mojo simulators for
the Wilson-Cowan E/I model. Same structure as bench_wong_wang.py."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PARAMS: dict[str, float] = dict(
    w_ee=10.0,
    w_ei=6.0,
    w_ie=10.0,
    w_ii=1.0,
    tau_e=1.0,
    tau_i=2.0,
    a=1.2,
    theta=4.0,
    dt=0.1,
)

N_STEPS = 100_000
PARITY_N = 3_000


def _probe_rust():
    try:
        from sc_neurocore_engine import py_wilson_cowan_simulate  # type: ignore

        return py_wilson_cowan_simulate, "available"
    except ImportError as e:
        return None, f"missing: {e}"


def _probe_julia():
    try:
        from sc_neurocore.accel.julia.neurons import (  # type: ignore
            _HAS_JULIA_NEURONS,
            simulate_wilson_cowan,
        )

        if not _HAS_JULIA_NEURONS:
            return None, "juliacall import failed"
        return simulate_wilson_cowan, "available"
    except ImportError as e:
        return None, f"missing: {e}"


def _probe_go():
    try:
        from sc_neurocore.accel.go.wilson_cowan import (  # type: ignore
            _HAS_GO_WILSON_COWAN,
            simulate_wilson_cowan,
        )

        if not _HAS_GO_WILSON_COWAN:
            return None, "libwilson_cowan.so not built (go build)"
        return simulate_wilson_cowan, "available"
    except ImportError as e:
        return None, f"missing: {e}"


def _probe_mojo():
    try:
        from sc_neurocore.accel.mojo.wilson_cowan import (  # type: ignore
            _HAS_MOJO_WILSON_COWAN,
            simulate_wilson_cowan,
        )

        if not _HAS_MOJO_WILSON_COWAN:
            return None, "libwilson_cowan.so not built (mojo build)"
        return simulate_wilson_cowan, "available"
    except ImportError as e:
        return None, f"missing: {e}"


def _run_python(n: int, ext: np.ndarray) -> float:
    u = WilsonCowanUnit(**DEFAULT_PARAMS)
    t0 = time.perf_counter()
    for t in range(n):
        u.step(float(ext[t]))
    return time.perf_counter() - t0


def _run_accel(fn: Callable, n: int, ext: np.ndarray):
    # Warm up
    fn(
        0.1,
        0.05,
        DEFAULT_PARAMS["w_ee"],
        DEFAULT_PARAMS["w_ei"],
        DEFAULT_PARAMS["w_ie"],
        DEFAULT_PARAMS["w_ii"],
        DEFAULT_PARAMS["tau_e"],
        DEFAULT_PARAMS["tau_i"],
        DEFAULT_PARAMS["a"],
        DEFAULT_PARAMS["theta"],
        DEFAULT_PARAMS["dt"],
        ext[:1_000],
    )
    t0 = time.perf_counter()
    out = fn(
        0.1,
        0.05,
        DEFAULT_PARAMS["w_ee"],
        DEFAULT_PARAMS["w_ei"],
        DEFAULT_PARAMS["w_ie"],
        DEFAULT_PARAMS["w_ii"],
        DEFAULT_PARAMS["tau_e"],
        DEFAULT_PARAMS["tau_i"],
        DEFAULT_PARAMS["a"],
        DEFAULT_PARAMS["theta"],
        DEFAULT_PARAMS["dt"],
        ext,
    )
    return time.perf_counter() - t0, out


def _parity(fn: Callable, n: int, ext: np.ndarray):
    return fn(
        0.1,
        0.05,
        DEFAULT_PARAMS["w_ee"],
        DEFAULT_PARAMS["w_ei"],
        DEFAULT_PARAMS["w_ie"],
        DEFAULT_PARAMS["w_ii"],
        DEFAULT_PARAMS["tau_e"],
        DEFAULT_PARAMS["tau_i"],
        DEFAULT_PARAMS["a"],
        DEFAULT_PARAMS["theta"],
        DEFAULT_PARAMS["dt"],
        ext,
    )


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
    ext = np.full(N_STEPS, 1.5, dtype=np.float64)
    backends: dict[str, dict[str, Any]] = {}

    t_py = _run_python(N_STEPS, ext)
    backends["python"] = {
        "available": True,
        "used": True,
        "reason": "primary reference",
        "wall_ms": round(t_py * 1e3, 3),
        "steps_per_s": round(N_STEPS / t_py, 0),
    }

    probes = [
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
        wall, out = _run_accel(fn, N_STEPS, ext)
        backends[name] = {
            "available": True,
            "used": True,
            "reason": reason,
            "wall_ms": round(wall * 1e3, 3),
            "steps_per_s": round(N_STEPS / wall, 0),
            "speedup_over_python": round(t_py / wall, 2),
        }
        outs[name] = out

    ref_name = "rust" if "rust" in outs else (next(iter(outs), None) if outs else None)
    parity: dict[str, Any] = {"reference": ref_name}
    if ref_name is not None:
        p_ext = np.linspace(-1.0, 4.0, PARITY_N)
        ref_fn = dict(probes)[ref_name]()[0]
        assert ref_fn is not None
        ref_out = _parity(ref_fn, PARITY_N, p_ext)
        for name, probe in probes:
            if name == ref_name:
                parity[name] = {"atol": 0.0}
                continue
            fn = backends[name].get("available") and probe()[0]
            if not fn:
                parity[name] = {"atol": "skipped"}
                continue
            out = _parity(fn, PARITY_N, p_ext)
            d_e = float(np.abs(ref_out["e"] - out["e"]).max())
            d_i = float(np.abs(ref_out["i"] - out["i"]).max())
            parity[name] = {
                "max_abs_delta_e": d_e,
                "max_abs_delta_i": d_i,
                "atol": max(d_e, d_i),
            }

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

    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": _cpu_model(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "n_steps": N_STEPS,
        "parity_n": PARITY_N,
        "params": DEFAULT_PARAMS,
    }
    out_path = RESULTS_DIR / "bench_wilson_cowan.json"
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "backends": backends, "parity": parity}, f, indent=2)
    print(f"\nResults → {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
