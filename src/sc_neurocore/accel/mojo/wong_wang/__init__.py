# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo-backed Wong-Wang batch (ctypes dispatch)

r"""Python entry point for the Mojo-compiled Wong-Wang batch simulator.

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

import numpy as np
import numpy.typing as npt

_LIB_PATH = Path(__file__).resolve().parent / "libwong_wang.so"


def _configure_library(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Attach the Wong-Wang ctypes signature to a loaded shared library."""
    lib.wong_wang_simulate_c.argtypes = [
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
    lib.wong_wang_simulate_c.restype = ctypes.c_int
    return lib


def _load_library() -> tuple[ctypes.CDLL | None, bool]:
    """Load the Mojo Wong-Wang shared library when it is available."""
    try:
        return _configure_library(ctypes.CDLL(str(_LIB_PATH))), True
    except OSError:
        return None, False


_lib, _HAS_MOJO_WONG_WANG = _load_library()


def _as_wong_wang_inputs(
    stim1: npt.ArrayLike,
    stim2: npt.ArrayLike,
    xi: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert and validate Wong-Wang input traces for ctypes dispatch."""
    stim1_arr = np.ascontiguousarray(stim1, dtype=np.float64)
    stim2_arr = np.ascontiguousarray(stim2, dtype=np.float64)
    xi_arr = np.ascontiguousarray(xi, dtype=np.float64)
    for name, array in (("stim1", stim1_arr), ("stim2", stim2_arr), ("xi", xi_arr)):
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional: got shape {array.shape}")
    n = stim1_arr.size
    if stim2_arr.size != n:
        raise ValueError(f"stim1 and stim2 length mismatch: {n} vs {stim2_arr.size}")
    if xi_arr.size != 2 * n:
        raise ValueError(f"xi length must be 2 * n_steps ({2 * n}): got {xi_arr.size}")
    return stim1_arr, stim2_arr, xi_arr


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
    stim1: npt.ArrayLike,
    stim2: npt.ArrayLike,
    xi: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64] | float]:
    """Run the Mojo-accelerated N-step Wong-Wang simulator.

    The stimulus and noise traces must be one-dimensional time-series. The
    wrapper validates their shapes before crossing the ctypes boundary so the
    Mojo shared library never receives implicitly flattened matrices. The
    return shape matches the Rust, Julia, and Go dispatchers.

    Mojo @export uses raw Int buffer addresses (parametric signature
    limitation in 0.26.2); the Mojo side reconstructs
    `UnsafePointer[Float64, MutAnyOrigin]` internally.
    """
    stim1_arr, stim2_arr, xi_arr = _as_wong_wang_inputs(stim1, stim2, xi)
    if _lib is None:
        raise ImportError(
            f"libwong_wang.so not built. Run: cd {_LIB_PATH.parent} && "
            f"~/.pixi/bin/mojo build --emit shared-lib -o {_LIB_PATH.name} wong_wang.mojo"
        )
    n = stim1_arr.size

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
        stim1_arr.ctypes.data,
        stim2_arr.ctypes.data,
        xi_arr.ctypes.data,
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
