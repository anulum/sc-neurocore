# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo-backed Wong-Wang batch (ctypes dispatch)

"""Python entry point for the Mojo-compiled Wong-Wang batch simulator.

Build:

    cd src/sc_neurocore/accel/mojo/wong_wang
    ~/.pixi/bin/mojo build --emit shared-lib \\
        -o libwong_wang.so wong_wang.mojo

The `.so` is platform-specific and gitignored; the `.mojo` source is
tracked. `_HAS_MOJO_WONG_WANG` flips True iff the lib is present.

Parity tolerance is intentionally numerical, not bit-pattern exact, for the
activation exponential. Mojo lowers `exp` through the host libm while the Rust
engine uses Rust `f64::exp`; both are IEEE-754 conforming implementations, but
the standard permits last-ULP differences after argument reduction and
polynomial approximation. The maintained parity tests therefore require tight
absolute agreement in the physical state variables rather than identical raw
floating-point bit patterns.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent / "libwong_wang.so"
_lib: ctypes.CDLL | None

try:
    _lib = ctypes.CDLL(str(_LIB_PATH))
    _lib.wong_wang_simulate_c.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _lib.wong_wang_simulate_c.restype = ctypes.c_int
    _HAS_MOJO_WONG_WANG = True
except OSError:
    _lib = None
    _HAS_MOJO_WONG_WANG = False


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
    stim1: np.ndarray[Any, Any] | list[float],
    stim2: np.ndarray[Any, Any] | list[float],
    xi: np.ndarray[Any, Any] | list[float],
) -> dict[str, Any]:
    """Mojo-accelerated N-step Wong-Wang simulator. Same signature + return
    shape as the Rust/Julia/Go dispatchers.

    Mojo @export uses raw Int buffer addresses (parametric signature
    limitation in 0.26.2); the Mojo side reconstructs
    `UnsafePointer[Float64, MutAnyOrigin]` internally.
    """
    if _lib is None:
        raise ImportError(
            f"libwong_wang.so not built. Run: cd {_LIB_PATH.parent} && "
            f"~/.pixi/bin/mojo build --emit shared-lib -o {_LIB_PATH.name} wong_wang.mojo"
        )
    stim1 = np.ascontiguousarray(stim1, dtype=np.float64)
    stim2 = np.ascontiguousarray(stim2, dtype=np.float64)
    xi = np.ascontiguousarray(xi, dtype=np.float64)
    n = stim1.size
    if stim2.size != n:
        raise ValueError(f"stim1 and stim2 length mismatch: {n} vs {stim2.size}")
    if xi.size != 2 * n:
        raise ValueError(f"xi length must be 2 * n_steps ({2 * n}): got {xi.size}")

    s1_out = np.empty(n, dtype=np.float64)
    s2_out = np.empty(n, dtype=np.float64)
    r1_out = np.empty(n, dtype=np.float64)
    r2_out = np.empty(n, dtype=np.float64)
    s1_final = np.zeros(1, dtype=np.float64)
    s2_final = np.zeros(1, dtype=np.float64)

    rc = _lib.wong_wang_simulate_c(
        ctypes.c_int(n),
        ctypes.c_double(s1_init),
        ctypes.c_double(s2_init),
        ctypes.c_double(tau_s),
        ctypes.c_double(gamma),
        ctypes.c_double(j_n),
        ctypes.c_double(j_cross),
        ctypes.c_double(i_0),
        ctypes.c_double(sigma),
        ctypes.c_double(dt),
        stim1.ctypes.data,
        stim2.ctypes.data,
        xi.ctypes.data,
        s1_out.ctypes.data,
        s2_out.ctypes.data,
        r1_out.ctypes.data,
        r2_out.ctypes.data,
        s1_final.ctypes.data,
        s2_final.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"Mojo wong_wang_simulate_c returned non-zero: {rc}")

    return {
        "s1": s1_out,
        "s2": s2_out,
        "r1": r1_out,
        "r2": r2_out,
        "s1_final": float(s1_final[0]),
        "s2_final": float(s2_final[0]),
    }
