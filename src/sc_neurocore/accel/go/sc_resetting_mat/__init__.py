# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ctypes facade for the Go SC resetting-MAT batch

"""Load the native Go project-model batch without fallback."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

_ACCEL_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY = _ACCEL_ROOT / "go" / "sc_resetting_mat" / "libsc_resetting_mat.so"
try:
    _lib: ctypes.CDLL | None = ctypes.CDLL(str(_LIBRARY))
    _function: Any | None = _lib.sc_resetting_mat_simulate_c
    _function.restype = ctypes.c_int
    _HAS_GO_SC_RESETTING_MAT = True
except (OSError, AttributeError):
    _lib = None
    _function = None
    _HAS_GO_SC_RESETTING_MAT = False


def simulate_sc_resetting_mat(*args: object) -> dict[str, object]:
    """Run the complete configured Go SC trace and enforce status zero."""
    if _function is None:
        raise RuntimeError("Go SC resetting-MAT shared library is unavailable")
    config = tuple(float(value) for value in args[:13])
    currents = np.ascontiguousarray(args[13], dtype=np.float64)
    if currents.ndim != 1:
        raise ValueError("Go SC resetting-MAT current must be one-dimensional")
    steps = currents.size
    traces = [np.empty(steps, dtype=np.float64) for _ in range(3)]
    events = np.empty(steps, dtype=np.int64)
    finals = [np.empty(1, dtype=np.float64) for _ in range(3)]
    code = _function(
        ctypes.c_int(steps),
        *(ctypes.c_double(value) for value in config),
        ctypes.c_void_p(currents.ctypes.data),
        *(ctypes.c_void_p(value.ctypes.data) for value in traces),
        ctypes.c_void_p(events.ctypes.data),
        *(ctypes.c_void_p(value.ctypes.data) for value in finals),
    )
    if code != 0:
        raise FloatingPointError(f"Go SC resetting-MAT batch failed with status {code}")
    return {
        "voltages": traces[0],
        "theta1": traces[1],
        "theta2": traces[2],
        "events": events,
        "v_final": float(finals[0][0]),
        "theta1_final": float(finals[1][0]),
        "theta2_final": float(finals[2][0]),
    }


__all__ = ["_HAS_GO_SC_RESETTING_MAT", "simulate_sc_resetting_mat"]
