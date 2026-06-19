# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go-backed Wilson-Cowan simulator (ctypes dispatch)

"""Python entry point for the Go-compiled Wilson-Cowan N-step simulator.

Build:

    cd src/sc_neurocore/accel/go/wilson_cowan
    PATH=/usr/local/go/bin:$PATH GOTOOLCHAIN=local \\
        go build -buildmode=c-shared -o libwilson_cowan.so wilson_cowan.go

The `.so` is platform-specific and gitignored; `.go` source + generated
`.h` header are tracked.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent / "libwilson_cowan.so"
_lib: ctypes.CDLL | None

try:
    _lib = ctypes.CDLL(str(_LIB_PATH))
    _lib.wilson_cowan_simulate_c.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,  # e_init, i_init
        ctypes.c_double,
        ctypes.c_double,  # w_ee, w_ei
        ctypes.c_double,
        ctypes.c_double,  # w_ie, w_ii
        ctypes.c_double,
        ctypes.c_double,  # tau_e, tau_i
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,  # a, theta, dt
        ctypes.c_void_p,  # ext ptr
        ctypes.c_void_p,
        ctypes.c_void_p,  # e_out, i_out
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),  # final
    ]
    _lib.wilson_cowan_simulate_c.restype = ctypes.c_int
    _HAS_GO_WILSON_COWAN = True
except OSError:
    _lib = None
    _HAS_GO_WILSON_COWAN = False


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
    ext_input: np.ndarray[Any, Any] | list[float],
) -> dict[str, Any]:
    """Go-accelerated Wilson-Cowan N-step simulator."""
    if _lib is None:
        raise ImportError(
            f"libwilson_cowan.so not built. Run: cd {_LIB_PATH.parent} && "
            f"go build -buildmode=c-shared -o {_LIB_PATH.name} wilson_cowan.go"
        )
    ext = np.ascontiguousarray(ext_input, dtype=np.float64)
    n = ext.size
    e_out = np.empty(n, dtype=np.float64)
    i_out = np.empty(n, dtype=np.float64)
    e_final = ctypes.c_double(0.0)
    i_final = ctypes.c_double(0.0)
    rc = _lib.wilson_cowan_simulate_c(
        ctypes.c_int(n),
        ctypes.c_double(e_init),
        ctypes.c_double(i_init),
        ctypes.c_double(w_ee),
        ctypes.c_double(w_ei),
        ctypes.c_double(w_ie),
        ctypes.c_double(w_ii),
        ctypes.c_double(tau_e),
        ctypes.c_double(tau_i),
        ctypes.c_double(a),
        ctypes.c_double(theta),
        ctypes.c_double(dt),
        ext.ctypes.data,
        e_out.ctypes.data,
        i_out.ctypes.data,
        ctypes.byref(e_final),
        ctypes.byref(i_final),
    )
    if rc != 0:
        raise RuntimeError(f"wilson_cowan_simulate_c returned non-zero: {rc}")
    return {
        "e": e_out,
        "i": i_out,
        "e_final": float(e_final.value),
        "i_final": float(i_final.value),
    }
