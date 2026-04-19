# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia-backed neuron kernels (juliacall dispatch)

"""Python entry points for Julia neuron kernels under this directory.

Each helper below lazily boots Julia via ``juliacall`` the first time it
is called and caches the compiled module so subsequent calls skip the
JIT warm-up (~5-10 s on cold start; sub-millisecond warm).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from juliacall import Main as _jl  # type: ignore[import-untyped,import-not-found]

    _HAS_JULIA_NEURONS = True
except ImportError:
    _jl = None  # type: ignore[assignment]
    _HAS_JULIA_NEURONS = False


_KERNEL_DIR = Path(__file__).resolve().parent
_WONG_WANG_LOADED = False
_WILSON_COWAN_LOADED = False


def _ensure_wong_wang_loaded() -> Any:
    """Include `wong_wang.jl` into Julia Main on first use; return the module."""
    global _WONG_WANG_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _WONG_WANG_LOADED:
        jl_path = _KERNEL_DIR / "wong_wang.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"wong_wang.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _WONG_WANG_LOADED = True
    return _jl.WongWangAccel


def _ensure_wilson_cowan_loaded() -> Any:
    """Include `wilson_cowan.jl` into Julia Main on first use; return the module."""
    global _WILSON_COWAN_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _WILSON_COWAN_LOADED:
        jl_path = _KERNEL_DIR / "wilson_cowan.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"wilson_cowan.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _WILSON_COWAN_LOADED = True
    return _jl.WilsonCowanAccel


def simulate_wong_wang(
    s1_init: float,
    s2_init: float,
    tau_s: float,
    gamma: float,
    j_n: float,
    j_cross: float,
    i_0: float,
    sigma: float,
    dt: float,
    stim1: np.ndarray | list[float],
    stim2: np.ndarray | list[float],
    xi: np.ndarray | list[float],
) -> dict[str, Any]:
    """Julia-accelerated N-step Wong-Wang simulator; parity with
    ``sc_neurocore_engine.py_wong_wang_simulate``. Returns a dict with
    per-step ``s1``/``s2``/``r1``/``r2`` arrays + final scalars.
    """
    mod = _ensure_wong_wang_loaded()
    stim1 = np.asarray(stim1, dtype=np.float64)
    stim2 = np.asarray(stim2, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    n = stim1.size
    if stim2.size != n:
        raise ValueError(f"stim1 and stim2 length mismatch: {n} vs {stim2.size}")
    if xi.size != 2 * n:
        raise ValueError(f"xi length must be 2 * n_steps ({2 * n}): got {xi.size}")
    s1_out = np.empty(n, dtype=np.float64)
    s2_out = np.empty(n, dtype=np.float64)
    r1_out = np.empty(n, dtype=np.float64)
    r2_out = np.empty(n, dtype=np.float64)
    s1_final, s2_final = mod.simulate_wong_wang_b(
        s1_init,
        s2_init,
        tau_s,
        gamma,
        j_n,
        j_cross,
        i_0,
        sigma,
        dt,
        stim1,
        stim2,
        xi,
        s1_out,
        s2_out,
        r1_out,
        r2_out,
    )
    return {
        "s1": s1_out,
        "s2": s2_out,
        "r1": r1_out,
        "r2": r2_out,
        "s1_final": float(s1_final),
        "s2_final": float(s2_final),
    }


def simulate_wilson_cowan(
    e_init: float,
    i_init: float,
    w_ee: float,
    w_ei: float,
    w_ie: float,
    w_ii: float,
    tau_e: float,
    tau_i: float,
    a: float,
    theta: float,
    dt: float,
    ext_input: np.ndarray | list[float],
) -> dict[str, Any]:
    """Julia-accelerated N-step Wilson-Cowan simulator; parity with
    ``sc_neurocore_engine.py_wilson_cowan_simulate``. Returns a dict
    with per-step ``e``/``i`` arrays + final scalars.
    """
    mod = _ensure_wilson_cowan_loaded()
    ext_input = np.asarray(ext_input, dtype=np.float64)
    n = ext_input.size
    e_out = np.empty(n, dtype=np.float64)
    i_out = np.empty(n, dtype=np.float64)
    e_final, i_final = mod.simulate_wilson_cowan_b(
        e_init,
        i_init,
        w_ee,
        w_ei,
        w_ie,
        w_ii,
        tau_e,
        tau_i,
        a,
        theta,
        dt,
        ext_input,
        e_out,
        i_out,
    )
    return {
        "e": e_out,
        "i": i_out,
        "e_final": float(e_final),
        "i_final": float(i_final),
    }
