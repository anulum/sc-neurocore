# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ctypes facade for the Mojo source MAT* batch

"""Load the executable Mojo MAT* ABI without surrogate substitution."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, cast

import numpy as np

_ACCEL_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY = _ACCEL_ROOT / "mojo" / "mat" / "libmat.so"
try:
    _lib = ctypes.CDLL(str(_LIBRARY))
    function: Any = _lib.mat_simulate_c
    function.restype = ctypes.c_int
    _function: Any | None = function
    _HAS_MOJO_MAT = True
except (OSError, AttributeError):
    _function = None
    _HAS_MOJO_MAT = False


def simulate_mat(*args: object) -> dict[str, object]:
    """Run a complete Mojo MAT* batch and expose only status-zero output."""
    if _function is None:
        raise RuntimeError("Mojo MAT shared library is unavailable")
    config = tuple(float(cast(float | int, value)) for value in args[:13])
    currents = np.ascontiguousarray(args[13], dtype=np.float64)
    if currents.ndim != 1:
        raise ValueError("Mojo MAT current must be one-dimensional")
    steps = currents.size
    traces = [np.empty(steps, dtype=np.float64) for _ in range(4)]
    events = np.empty(steps, dtype=np.int64)
    finals = [np.empty(1, dtype=np.float64) for _ in range(4)]
    code = _function(
        ctypes.c_ssize_t(steps),
        *(ctypes.c_double(value) for value in config),
        ctypes.c_ssize_t(currents.ctypes.data),
        *(ctypes.c_ssize_t(value.ctypes.data) for value in traces),
        ctypes.c_ssize_t(events.ctypes.data),
        *(ctypes.c_ssize_t(value.ctypes.data) for value in finals),
    )
    if code != 0:
        raise FloatingPointError(f"Mojo MAT batch failed with status {code}")
    return {
        "voltages": traces[0],
        "theta1": traces[1],
        "theta2": traces[2],
        "refractory": traces[3],
        "events": events,
        "v_final": float(finals[0][0]),
        "theta1_final": float(finals[1][0]),
        "theta2_final": float(finals[2][0]),
        "refractory_final": float(finals[3][0]),
    }


__all__ = ["_HAS_MOJO_MAT", "simulate_mat"]
