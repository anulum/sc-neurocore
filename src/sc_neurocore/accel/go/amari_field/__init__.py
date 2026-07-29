# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ctypes facade for the Go Amari field batch

"""Load and validate the Go C-shared Amari neural-field implementation."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
_LIBRARY = Path(__file__).with_name("libamari_field.so")
_LIB: ctypes.CDLL | None
_FUNCTION: Any | None
try:
    _LIB = ctypes.CDLL(str(_LIBRARY))
    _FUNCTION = _LIB.amari_field_simulate_c
    _FUNCTION.restype = ctypes.c_int
    _HAS_GO_AMARI_FIELD = True
except (OSError, AttributeError):
    _LIB = None
    _FUNCTION = None
    _HAS_GO_AMARI_FIELD = False


def simulate_amari_field(
    u_init: FloatArray,
    tau: float,
    a_exc: float,
    a_width: float,
    b_inh: float,
    b_width: float,
    dx: float,
    dt: float,
    currents: FloatArray,
) -> dict[str, object]:
    """Run the complete Go vector batch or raise on any native error code."""
    if _FUNCTION is None:
        raise RuntimeError("Go Amari field shared library is unavailable")
    steps, n = currents.shape
    states = np.empty((steps, n), dtype=np.float64)
    rates = np.empty(steps, dtype=np.float64)
    final = np.empty(n, dtype=np.float64)
    pointer = ctypes.c_void_p
    code = _FUNCTION(
        ctypes.c_int(steps),
        ctypes.c_int(n),
        ctypes.c_double(tau),
        ctypes.c_double(a_exc),
        ctypes.c_double(a_width),
        ctypes.c_double(b_inh),
        ctypes.c_double(b_width),
        ctypes.c_double(dx),
        ctypes.c_double(dt),
        pointer(u_init.ctypes.data),
        pointer(currents.ctypes.data),
        pointer(states.ctypes.data),
        pointer(rates.ctypes.data),
        pointer(final.ctypes.data),
    )
    if code != 0:
        raise FloatingPointError(f"Go Amari field batch failed with status {code}")
    return {"states": states, "mean_rates": rates, "final_state": final}


__all__ = ["_HAS_GO_AMARI_FIELD", "simulate_amari_field"]
