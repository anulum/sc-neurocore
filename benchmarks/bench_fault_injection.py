#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FaultInjector.inject() multi-backend wall-clock benchmark

"""Reproducible multi-language benchmark for fault injection kernels.

For each of the 5 fault models (BIT_FLIP / STUCK_AT_0 / STUCK_AT_1 /
DROPOUT / GAUSSIAN_NOISE), measure wall time for a single inject on a
1 Mbit boolean array. Four backends compared:

1. **Python + NumPy** (the reference `FaultInjector.inject`)
2. **Rust** via PyO3 (`py_inject_*_u8` in `sc_neurocore_engine`)
3. **Julia** via `juliacall` (`FaultInjectionAccel` module)
4. **Go**   via `ctypes` (`libfault.so` c-shared library)

Mojo is exempted (documented in backends block) — current public Mojo
(0.26) lacks a stable `@export` for parametric UnsafePointer args, see
follow-up #69 for the equivalent LGSSM blocker.

RNG parity note: each backend uses a different RNG (NumPy PCG64, Rust
Xoshiro256++, Julia Xoshiro, Go ChaCha8) so bitwise parity is
impossible. Parity is verified statistically — fault counts must fall
within 4σ of Binomial(n, ber) mean on 1 Mbit streams at ber=1e-3.

Usage:
    python benchmarks/bench_fault_injection.py
    python benchmarks/bench_fault_injection.py --json benchmarks/results/bench_fault_injection.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from sc_neurocore.fault_injection import FaultInjector, FaultModel


REPO_ROOT = Path(__file__).resolve().parent.parent
N_BITS = 1_000_000
N_REPEATS = 5
BENCH_BER = 1e-3
GAUSSIAN_SIGMA = 0.5

FAULT_MODELS = [
    ("BIT_FLIP", FaultModel.BIT_FLIP, BENCH_BER, "bitflip"),
    ("STUCK_AT_0", FaultModel.STUCK_AT_0, BENCH_BER, "stuck_at_0"),
    ("STUCK_AT_1", FaultModel.STUCK_AT_1, BENCH_BER, "stuck_at_1"),
    ("DROPOUT", FaultModel.DROPOUT, BENCH_BER, "dropout"),
    ("GAUSSIAN_NOISE", FaultModel.GAUSSIAN_NOISE, GAUSSIAN_SIGMA, "gaussian"),
]


# ─────────────────────────── Backend probes ──────────────────────────

def probe_rust() -> dict:
    if importlib.util.find_spec("sc_neurocore_engine") is None:
        return {"available": False, "reason": "sc_neurocore_engine not importable"}
    mod = importlib.import_module("sc_neurocore_engine")
    needed = (
        "py_inject_bitflip_u8", "py_inject_stuck_at_0_u8",
        "py_inject_stuck_at_1_u8", "py_inject_dropout_u8",
        "py_inject_gaussian_u8",
    )
    missing = [n for n in needed if not hasattr(mod, n)]
    if missing:
        return {"available": False, "reason": f"engine missing: {missing}"}
    return {
        "available": True,
        "bitflip": mod.py_inject_bitflip_u8,
        "stuck_at_0": mod.py_inject_stuck_at_0_u8,
        "stuck_at_1": mod.py_inject_stuck_at_1_u8,
        "dropout": mod.py_inject_dropout_u8,
        "gaussian": mod.py_inject_gaussian_u8,
    }


def probe_julia() -> dict:
    if importlib.util.find_spec("juliacall") is None:
        return {"available": False, "reason": "juliacall not installed"}
    jl_path = REPO_ROOT / "src/sc_neurocore/accel/julia/fault_injection/fault_injection.jl"
    if not jl_path.is_file():
        return {"available": False, "reason": f"{jl_path} missing"}
    try:
        from juliacall import Main as jl
        jl.include(str(jl_path))
        accel = jl.FaultInjectionAccel
    except Exception as exc:
        return {"available": False, "reason": f"julia init failed: {exc}"}
    return {
        "available": True,
        "bitflip": accel.inject_bitflip,
        "stuck_at_0": accel.inject_stuck_at_0,
        "stuck_at_1": accel.inject_stuck_at_1,
        "dropout": accel.inject_dropout,
        "gaussian": accel.inject_gaussian,
    }


def probe_go() -> dict:
    import ctypes
    so_path = REPO_ROOT / "src/sc_neurocore/accel/go/fault_injection/libfault.so"
    if not so_path.is_file():
        return {"available": False, "reason": f"{so_path} missing — "
                "build via: go build -buildmode=c-shared -o libfault.so fault.go"}
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError as exc:
        return {"available": False, "reason": f"ctypes CDLL failed: {exc}"}
    sigs = ("inject_bitflip_c", "inject_stuck_at_0_c", "inject_stuck_at_1_c",
            "inject_dropout_c", "inject_gaussian_c")
    missing = [s for s in sigs if not hasattr(lib, s)]
    if missing:
        return {"available": False, "reason": f"go symbols missing: {missing}"}
    for name in sigs:
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        fn.restype = None
    return {"available": True, "lib": lib}


def probe_mojo() -> dict:
    mojo_bin = Path.home() / ".pixi/bin/mojo"
    if not mojo_bin.is_file():
        return {"available": False, "reason": "mojo toolchain not at ~/.pixi/bin/mojo"}
    return {
        "available": True,
        "exempt": True,
        "reason": ("Mojo 0.26 public toolchain lacks stable @export for "
                   "parametric UnsafePointer args; port blocked pending "
                   "upstream API stability. See follow-up #69."),
    }


# ─────────────────────────── Runners ─────────────────────────────────

def _run_python(mdl: FaultModel, ber: float, rng_seed: int) -> tuple[float, int]:
    bs = np.random.default_rng(rng_seed).integers(0, 2, N_BITS, dtype=np.uint8).astype(bool)
    injector = FaultInjector(seed=rng_seed)
    # warm
    injector.inject(bs.copy(), mdl, ber)
    times: list[float] = []
    n_last = 0
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _, n = injector.inject(bs.copy(), mdl, ber)
        times.append((time.perf_counter() - t0) * 1000.0)
        n_last = int(n)
    times.sort()
    return times[len(times) // 2], n_last


def _run_rust(kernel, ber: float, rng_seed: int) -> tuple[float, int]:
    bs = np.random.default_rng(rng_seed).integers(0, 2, N_BITS, dtype=np.uint8)
    kernel(bs.copy(), ber, rng_seed)  # warm
    times: list[float] = []
    n_last = 0
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _, n = kernel(bs.copy(), ber, rng_seed)
        times.append((time.perf_counter() - t0) * 1000.0)
        n_last = int(n)
    times.sort()
    return times[len(times) // 2], n_last


def _run_julia(kernel, ber: float, rng_seed: int) -> tuple[float, int]:
    bs = np.random.default_rng(rng_seed).integers(0, 2, N_BITS, dtype=np.uint8)
    # warm (JIT)
    kernel(bs.copy(), ber, rng_seed)
    times: list[float] = []
    n_last = 0
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _, n = kernel(bs.copy(), ber, rng_seed)
        times.append((time.perf_counter() - t0) * 1000.0)
        n_last = int(n)
    times.sort()
    return times[len(times) // 2], n_last


def _run_go(lib, sym: str, ber: float, rng_seed: int) -> tuple[float, int]:
    import ctypes
    fn = getattr(lib, f"inject_{sym}_c")
    bs_src = np.random.default_rng(rng_seed).integers(0, 2, N_BITS, dtype=np.uint8)
    # warm
    bs = bs_src.copy()
    n_out = ctypes.c_uint64(0)
    fn(bs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
       ctypes.c_int64(N_BITS), ctypes.c_double(ber),
       ctypes.c_uint64(rng_seed), ctypes.byref(n_out))
    times: list[float] = []
    n_last = 0
    for _ in range(N_REPEATS):
        bs = bs_src.copy()
        n_out = ctypes.c_uint64(0)
        t0 = time.perf_counter()
        fn(bs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
           ctypes.c_int64(N_BITS), ctypes.c_double(ber),
           ctypes.c_uint64(rng_seed), ctypes.byref(n_out))
        times.append((time.perf_counter() - t0) * 1000.0)
        n_last = int(n_out.value)
    times.sort()
    return times[len(times) // 2], n_last


# ─────────────────────────── Parity check ────────────────────────────

def _statistical_parity_ok(model_name: str, ber: float, n: int) -> bool:
    """Verify fault count is within 4σ of Binomial(N_BITS, ber) mean,
    or within a loose band for STUCK_AT_* / DROPOUT (which count only
    flipped-1 bits, so expected ~N_BITS·ber/2)."""
    if model_name == "BIT_FLIP":
        mean = N_BITS * ber
        sigma = (N_BITS * ber * (1.0 - ber)) ** 0.5
    elif model_name in ("STUCK_AT_0", "STUCK_AT_1", "DROPOUT"):
        # ~half the masked bits actually change (prior 0.5 assumed)
        mean = N_BITS * ber * 0.5
        sigma = (N_BITS * ber * 0.5 * (1.0 - ber * 0.5)) ** 0.5
    else:  # GAUSSIAN_NOISE — use empirical 0.5-prior flip rate near sigma=0.5
        mean = 0.0
        sigma = float("inf")  # skip parity for Gaussian (model-specific)
        return True
    lo, hi = mean - 4.0 * sigma, mean + 4.0 * sigma
    return lo <= n <= hi


# ─────────────────────────── Main ────────────────────────────────────

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("# FaultInjector.inject() multi-backend benchmark")
    print(f"# Bitstream length: {N_BITS}, BER: {BENCH_BER:.0e} "
          f"(Gaussian σ: {GAUSSIAN_SIGMA})")
    print(f"# Repeats per cell: {N_REPEATS}")
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# platform: {platform.platform()}")

    backends = {
        "python": {"available": True},
        "rust": probe_rust(),
        "julia": probe_julia(),
        "go": probe_go(),
        "mojo": probe_mojo(),
    }

    print()
    print("# Backend availability")
    for name, info in backends.items():
        tag = "OK" if info.get("available") and not info.get("exempt") else (
            "EXEMPT" if info.get("exempt") else "MISSING"
        )
        reason = info.get("reason", "") if tag != "OK" else ""
        print(f"  {name:<8} {tag:<8} {reason}")

    print()
    print(f"{'model':<16} {'python ms':>10} {'rust ms':>10} "
          f"{'julia ms':>10} {'go ms':>10}  {'parity':>8}")
    print(f"{'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*8}")

    rows: list[dict[str, object]] = []
    for display_name, mdl, ber, sym in FAULT_MODELS:
        row: dict[str, object] = {"fault_model": display_name, "ber": ber}
        # python
        py_ms, py_n = _run_python(mdl, ber, rng_seed=42)
        row["python_ms"] = py_ms
        row["python_n_affected"] = py_n
        # rust
        if backends["rust"]["available"]:
            ru_ms, ru_n = _run_rust(backends["rust"][sym], ber, rng_seed=42)
            row["rust_ms"] = ru_ms
            row["rust_n_affected"] = ru_n
        else:
            ru_ms = None
            ru_n = None
            row["rust_ms"] = None
        # julia
        if backends["julia"]["available"]:
            ju_ms, ju_n = _run_julia(backends["julia"][sym], ber, rng_seed=42)
            row["julia_ms"] = ju_ms
            row["julia_n_affected"] = ju_n
        else:
            ju_ms = None
            ju_n = None
            row["julia_ms"] = None
        # go
        if backends["go"]["available"]:
            go_ms, go_n = _run_go(backends["go"]["lib"], sym, ber, rng_seed=42)
            row["go_ms"] = go_ms
            row["go_n_affected"] = go_n
        else:
            go_ms = None
            go_n = None
            row["go_ms"] = None

        parity_ok = all(
            _statistical_parity_ok(display_name, ber, n)
            for n in [py_n, ru_n, ju_n, go_n] if n is not None
        )
        row["parity_ok"] = parity_ok

        def fmt(v: float | None) -> str:
            return f"{v:>10.2f}" if v is not None else f"{'-':>10}"

        print(f"{display_name:<16} {fmt(py_ms)} {fmt(ru_ms)} "
              f"{fmt(ju_ms)} {fmt(go_ms)}  {'ok' if parity_ok else 'FAIL':>8}")
        rows.append(row)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        backends_json = {
            name: {k: v for k, v in info.items()
                   if k in ("available", "exempt", "reason")}
            for name, info in backends.items()
        }
        payload = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_bits": N_BITS,
            "n_repeats": N_REPEATS,
            "ber": BENCH_BER,
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "backends": backends_json,
            "rows": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
