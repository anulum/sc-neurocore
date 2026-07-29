# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ctypes facade for the Mojo Amari field batch

"""Load and validate the executable Mojo Amari neural-field implementation."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_LIBRARY = Path(__file__).with_name("libamari_field.so")
_LIB: ctypes.CDLL | None
_FUNCTION: Any | None
try:
    _LIB = ctypes.CDLL(str(_LIBRARY))
    _FUNCTION = _LIB.amari_field_simulate_c
    _FUNCTION.restype = ctypes.c_int
    _HAS_MOJO_AMARI_FIELD = True
except (OSError, AttributeError):
    _LIB = None
    _FUNCTION = None
    _HAS_MOJO_AMARI_FIELD = False


def simulate_amari_field(
    u_init: npt.NDArray[np.float64],
    tau: float,
    a_exc: float,
    a_width: float,
    b_inh: float,
    b_width: float,
    dx: float,
    dt: float,
    currents: npt.NDArray[np.float64],
) -> dict[str, object]:
    """Run the complete Mojo vector batch or raise on a nonzero status."""
    if _FUNCTION is None:
        raise RuntimeError("Mojo Amari field shared library is unavailable")
    steps, n = currents.shape
    states = np.empty((steps, n), dtype=np.float64)
    rates = np.empty(steps, dtype=np.float64)
    final = np.empty(n, dtype=np.float64)
    code = _FUNCTION(
        ctypes.c_ssize_t(steps),
        ctypes.c_ssize_t(n),
        ctypes.c_double(tau),
        ctypes.c_double(a_exc),
        ctypes.c_double(a_width),
        ctypes.c_double(b_inh),
        ctypes.c_double(b_width),
        ctypes.c_double(dx),
        ctypes.c_double(dt),
        ctypes.c_ssize_t(u_init.ctypes.data),
        ctypes.c_ssize_t(currents.ctypes.data),
        ctypes.c_ssize_t(states.ctypes.data),
        ctypes.c_ssize_t(rates.ctypes.data),
        ctypes.c_ssize_t(final.ctypes.data),
    )
    if code != 0:
        raise FloatingPointError(f"Mojo Amari field batch failed with status {code}")
    return {"states": states, "mean_rates": rates, "final_state": final}


__all__ = ["_HAS_MOJO_AMARI_FIELD", "simulate_amari_field"]
