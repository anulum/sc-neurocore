# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go-backed Wong-Wang batch (ctypes dispatch)

r"""Python entry point for the Go-compiled Wong-Wang batch simulator.

The Go source `wong_wang.go` must be pre-built into `libwong_wang.so`
via:

    cd src/sc_neurocore/accel/go/wong_wang
    PATH=/usr/local/go/bin:$PATH GOTOOLCHAIN=local \\
        go build -buildmode=c-shared -o libwong_wang.so wong_wang.go

The `.so` is platform-specific and gitignored; the `.go` source and
generated `.h` header are tracked. `_HAS_GO_WONG_WANG` is `True` iff
the shared lib exists next to this module at import time.
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
        ctypes.c_double,  # s1_init, s2_init
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,  # tau_s, gamma, j_n
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,  # j_cross, i_0, sigma
        ctypes.c_double,  # dt
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,  # stim1, stim2, xi
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,  # s1,s2,r1,r2 out
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),  # s1/2_final_out
    ]
    lib.wong_wang_simulate_c.restype = ctypes.c_int
    return lib


def _load_library() -> tuple[ctypes.CDLL | None, bool]:
    """Load the Go Wong-Wang shared library when it is available."""
    try:
        return _configure_library(ctypes.CDLL(str(_LIB_PATH))), True
    except OSError:
        return None, False


_lib, _HAS_GO_WONG_WANG = _load_library()


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
    """Run the Go-accelerated N-step Wong-Wang simulator.

    The stimulus and noise traces must be one-dimensional time-series. The
    wrapper validates their shapes before crossing the ctypes boundary so the
    Go shared library never receives implicitly flattened matrices. The return
    shape matches ``sc_neurocore_engine.py_wong_wang_simulate`` and the Julia
    dispatcher.
    """
    stim1_arr, stim2_arr, xi_arr = _as_wong_wang_inputs(stim1, stim2, xi)
    if _lib is None:
        raise ImportError(
            f"libwong_wang.so not built. Run: cd {_LIB_PATH.parent} && "
            f"go build -buildmode=c-shared -o {_LIB_PATH.name} wong_wang.go"
        )
    n = stim1_arr.size

    s1_out = np.empty(n, dtype=np.float64)
    s2_out = np.empty(n, dtype=np.float64)
    r1_out = np.empty(n, dtype=np.float64)
    r2_out = np.empty(n, dtype=np.float64)
    s1_final = ctypes.c_double(0.0)
    s2_final = ctypes.c_double(0.0)

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
        ctypes.byref(s1_final),
        ctypes.byref(s2_final),
    )
    if rc != 0:
        raise RuntimeError(f"wong_wang_simulate_c returned non-zero: {rc}")

    return {
        "s1": s1_out,
        "s2": s2_out,
        "r1": r1_out,
        "r2": r2_out,
        "s1_final": float(s1_final.value),
        "s2_final": float(s2_final.value),
    }
