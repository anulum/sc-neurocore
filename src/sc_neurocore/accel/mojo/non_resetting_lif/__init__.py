# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Load the executable Mojo MAT(1) ABI without fallback substitution."""

from __future__ import annotations
import ctypes
from pathlib import Path
from typing import Any, cast
import numpy as np

_LIBRARY = Path(__file__).with_name("libnon_resetting_lif.so")
try:
    _lib = ctypes.CDLL(str(_LIBRARY))
    function: Any = _lib.non_resetting_lif_simulate_c
    function.restype = ctypes.c_int
    _function: Any | None = function
    _HAS_MOJO_NON_RESETTING_LIF = True
except (OSError, AttributeError):
    _function = None
    _HAS_MOJO_NON_RESETTING_LIF = False


def simulate_non_resetting_lif(*args: object) -> dict[str, object]:
    """Run the complete configured Mojo MAT(1) trace."""
    if _function is None:
        raise RuntimeError("Mojo MAT(1) shared library is unavailable")
    config = tuple(float(cast(float | int, value)) for value in args[:10])
    currents = np.ascontiguousarray(args[10], dtype=np.float64)
    if currents.ndim != 1:
        raise ValueError("Mojo MAT(1) current must be one-dimensional")
    traces = [np.empty(currents.size, dtype=np.float64) for _ in range(3)]
    events = np.empty(currents.size, dtype=np.int64)
    finals = [np.empty(1, dtype=np.float64) for _ in range(3)]
    code = _function(
        ctypes.c_ssize_t(currents.size),
        *(ctypes.c_double(value) for value in config),
        ctypes.c_ssize_t(currents.ctypes.data),
        *(ctypes.c_ssize_t(value.ctypes.data) for value in traces),
        ctypes.c_ssize_t(events.ctypes.data),
        *(ctypes.c_ssize_t(value.ctypes.data) for value in finals),
    )
    if code != 0:
        raise FloatingPointError(f"Mojo MAT(1) batch failed with status {code}")
    return {
        "voltages": traces[0],
        "theta": traces[1],
        "refractory": traces[2],
        "events": events,
        "v_final": float(finals[0][0]),
        "theta_final": float(finals[1][0]),
        "refractory_final": float(finals[2][0]),
    }


__all__ = ["_HAS_MOJO_NON_RESETTING_LIF", "simulate_non_resetting_lif"]
