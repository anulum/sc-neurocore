# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo-backed Wilson-Cowan simulator (ctypes dispatch)

r"""Python entry point for the Mojo-compiled Wilson-Cowan N-step simulator.

Build:

    cd src/sc_neurocore/accel/mojo/wilson_cowan
    ~/.pixi/bin/mojo build --emit shared-lib \\
        -o libwilson_cowan.so wilson_cowan.mojo

The `.so` is platform-specific and gitignored; the `.mojo` source is
tracked.

The Wilson-Cowan transfer function uses the same branch-stable logistic as the
Python reference, so its exponential is evaluated only at non-positive
arguments even at finite saturation inputs.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt

_LIB_PATH = Path(__file__).resolve().parent / "libwilson_cowan.so"
_MAX_STEPS = (1 << 31) - 1


def _configure_library(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Attach the Wilson-Cowan ctypes signature to a loaded shared library."""
    lib.wilson_cowan_simulate_c.argtypes = [
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
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.wilson_cowan_simulate_c.restype = ctypes.c_int
    return lib


def _load_library() -> tuple[ctypes.CDLL | None, bool]:
    """Load the Mojo Wilson-Cowan shared library when it is available."""
    try:
        return _configure_library(ctypes.CDLL(str(_LIB_PATH))), True
    except OSError:
        return None, False


_lib, _HAS_MOJO_WILSON_COWAN = _load_library()


def _as_ext_input(ext_input: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert the external drive into a contiguous one-dimensional vector."""
    ext = np.ascontiguousarray(ext_input, dtype=np.float64)
    if ext.ndim != 1:
        raise ValueError(f"ext_input must be one-dimensional: got shape {ext.shape}")
    if ext.size > _MAX_STEPS:
        raise ValueError(f"ext_input length must be at most {_MAX_STEPS}")
    if not np.isfinite(ext).all():
        raise ValueError("ext_input must contain only finite values")
    return ext


def _validate_configuration(
    e: float,
    i: float,
    w_ee: float,
    w_ei: float,
    w_ie: float,
    w_ii: float,
    tau_e: float,
    tau_i: float,
    a: float,
    theta: float,
    dt: float,
) -> None:
    """Reject unsafe scalars before crossing the exported Mojo boundary."""
    values = (e, i, w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Wilson-Cowan configuration must be finite")
    if any(weight < 0.0 for weight in (w_ee, w_ei, w_ie, w_ii)):
        raise ValueError("Wilson-Cowan weights must be non-negative")
    if tau_e <= 0.0 or tau_i <= 0.0 or a <= 0.0 or dt <= 0.0:
        raise ValueError("Wilson-Cowan time constants, gain, and dt must be positive")
    z = -a * theta
    exp_z = math.exp(z) if z < 0.0 else math.exp(-z)
    baseline = exp_z / (1.0 + exp_z) if z < 0.0 else 1.0 / (1.0 + exp_z)
    if not -baseline <= e <= 1.0 or not -baseline <= i <= 1.0:
        raise ValueError("Wilson-Cowan initial rates are outside the state envelope")


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
    ext_input: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64] | float]:
    """Run the Mojo-accelerated Wilson-Cowan N-step simulator.

    The external drive must be a one-dimensional time-series. The wrapper
    validates that shape before crossing the ctypes boundary so the Mojo shared
    library never receives an implicitly flattened matrix.
    """
    ext = _as_ext_input(ext_input)
    _validate_configuration(
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
    )
    if _lib is None:
        raise ImportError(
            f"libwilson_cowan.so not built. Run: cd {_LIB_PATH.parent} && "
            f"~/.pixi/bin/mojo build --emit shared-lib "
            f"-o {_LIB_PATH.name} wilson_cowan.mojo"
        )
    n = ext.size
    e_out = np.empty(n, dtype=np.float64)
    i_out = np.empty(n, dtype=np.float64)
    e_final = np.zeros(1, dtype=np.float64)
    i_final = np.zeros(1, dtype=np.float64)
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
        e_final.ctypes.data,
        i_final.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"Mojo wilson_cowan_simulate_c returned non-zero: {rc}")
    return {
        "e": e_out,
        "i": i_out,
        "e_final": float(e_final[0]),
        "i_final": float(i_final[0]),
    }
