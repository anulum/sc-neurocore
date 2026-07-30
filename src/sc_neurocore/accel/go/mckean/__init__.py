# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""ctypes loader for the source McKean Go ABI."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt


class _Out(ctypes.Structure):
    _fields_ = [
        ("v", ctypes.c_double),
        ("w", ctypes.c_double),
        ("event", ctypes.c_int32),
        ("status", ctypes.c_int32),
    ]


_LIB: ctypes.CDLL | None
try:
    _LIB = ctypes.CDLL(str(Path(__file__).with_name("libmckean.so")))
    _LIB.mckean_step.argtypes = [ctypes.c_double] * 8
    _LIB.mckean_step.restype = _Out
    _HAS_GO_MCKEAN = True
except OSError:
    _LIB = None
    _HAS_GO_MCKEAN = False


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
    """Execute a finite current trace through the Go C ABI."""
    if _LIB is None:
        raise RuntimeError("Go McKean shared library unavailable")
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    voltages = np.empty(drive.size)
    recovery = np.empty(drive.size)
    events = np.empty(drive.size, dtype=np.int64)
    for index, current in enumerate(drive):
        out = _LIB.mckean_step(v, w, a, lambda_, mu, b, dt, float(current))
        if out.status:
            raise ValueError("Go McKean transition failed")
        v, w = out.v, out.w
        voltages[index] = v
        recovery[index] = w
        events[index] = out.event
    return {
        "voltages": voltages,
        "recovery": recovery,
        "events": events,
        "v_final": v,
        "w_final": w,
    }
