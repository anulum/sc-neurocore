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
    from juliacall import Main as _jl  # type: ignore[import-not-found]

    _HAS_JULIA_NEURONS = True
except ImportError:
    _jl = None  # type: ignore[assignment]
    _HAS_JULIA_NEURONS = False


_KERNEL_DIR = Path(__file__).resolve().parent
_WONG_WANG_LOADED = False


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
    stim1,
    stim2,
    xi,
):
    """Julia-accelerated batch Wong-Wang simulator; parity with
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
