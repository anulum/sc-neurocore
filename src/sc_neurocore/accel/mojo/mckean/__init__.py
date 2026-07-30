# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Load the executable Mojo McKean ABI."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_fn: Any | None
try:
    _lib = ctypes.CDLL(str(Path(__file__).with_name("libmckean.so")))
    _fn = _lib.mckean_simulate_c
    _fn.restype = ctypes.c_int
    _HAS_MOJO_MCKEAN = True
except (OSError, AttributeError):
    _fn = None
    _HAS_MOJO_MCKEAN = False


def simulate_mckean(
    v: float,
    w: float,
    a: float,
    lambda_: float,
    mu: float,
    b: float,
    dt: float,
    currents: npt.ArrayLike,
) -> dict[str, object]:
    """Execute a finite current trace through the Mojo C ABI."""
    if _fn is None:
        raise RuntimeError("Mojo McKean shared library unavailable")
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    voltages = np.empty(drive.size)
    recovery = np.empty(drive.size)
    events = np.empty(drive.size, dtype=np.int64)
    v_final = np.empty(1)
    w_final = np.empty(1)
    code = _fn(
        ctypes.c_ssize_t(drive.size),
        *(ctypes.c_double(x) for x in (v, w, a, lambda_, mu, b, dt)),
        ctypes.c_ssize_t(drive.ctypes.data),
        ctypes.c_ssize_t(voltages.ctypes.data),
        ctypes.c_ssize_t(recovery.ctypes.data),
        ctypes.c_ssize_t(events.ctypes.data),
        ctypes.c_ssize_t(v_final.ctypes.data),
        ctypes.c_ssize_t(w_final.ctypes.data),
    )
    if code:
        raise FloatingPointError(f"Mojo McKean batch failed with status {code}")
    return {
        "voltages": voltages,
        "recovery": recovery,
        "events": events,
        "v_final": float(v_final[0]),
        "w_final": float(w_final[0]),
    }
