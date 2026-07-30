# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Load the native Go retained-project batch without fallback substitution."""

from __future__ import annotations
import ctypes
from pathlib import Path
from typing import Any
import numpy as np

_LIBRARY = Path(__file__).with_name("libsc_non_resetting_adaptive_lif.so")
try:
    _lib: ctypes.CDLL | None = ctypes.CDLL(str(_LIBRARY))
    _function: Any | None = _lib.sc_non_resetting_adaptive_lif_simulate_c
    _function.restype = ctypes.c_int
    _HAS_GO_SC_NON_RESETTING_ADAPTIVE_LIF = True
except (OSError, AttributeError):
    _lib = None
    _function = None
    _HAS_GO_SC_NON_RESETTING_ADAPTIVE_LIF = False


def simulate_sc_non_resetting_adaptive_lif(*args: object) -> dict[str, object]:
    """Run the complete configured Go retained-project trace."""
    if _function is None:
        raise RuntimeError("Go SC adaptive LIF shared library is unavailable")
    config = tuple(float(value) for value in args[:9])
    currents = np.ascontiguousarray(args[9], dtype=np.float64)
    if currents.ndim != 1:
        raise ValueError("Go SC adaptive LIF current must be one-dimensional")
    traces = [np.empty(currents.size, dtype=np.float64) for _ in range(2)]
    events = np.empty(currents.size, dtype=np.int64)
    finals = [np.empty(1, dtype=np.float64) for _ in range(2)]
    code = _function(
        ctypes.c_int(currents.size),
        *(ctypes.c_double(value) for value in config),
        ctypes.c_void_p(currents.ctypes.data),
        *(ctypes.c_void_p(value.ctypes.data) for value in traces),
        ctypes.c_void_p(events.ctypes.data),
        *(ctypes.c_void_p(value.ctypes.data) for value in finals),
    )
    if code != 0:
        raise FloatingPointError(f"Go SC adaptive LIF batch failed with status {code}")
    return {
        "voltages": traces[0],
        "theta": traces[1],
        "events": events,
        "v_final": float(finals[0][0]),
        "theta_final": float(finals[1][0]),
    }


__all__ = ["_HAS_GO_SC_NON_RESETTING_ADAPTIVE_LIF", "simulate_sc_non_resetting_adaptive_lif"]
