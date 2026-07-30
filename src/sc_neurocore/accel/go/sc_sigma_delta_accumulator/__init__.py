# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Load the native Go retained bipolar accumulator batch."""

from __future__ import annotations
import ctypes
from pathlib import Path
from typing import Any, cast
import numpy as np

_LIBRARY = Path(__file__).with_name("libsc_sigma_delta_accumulator.so")
try:
    _loaded = ctypes.CDLL(str(_LIBRARY))
    _loaded_function: Any = _loaded.sc_sigma_delta_accumulator_simulate_c
    _loaded_function.restype = ctypes.c_int
    _lib: ctypes.CDLL | None = _loaded
    _function: Any | None = _loaded_function
    _HAS_GO_SC_SIGMA_DELTA_ACCUMULATOR = True
except (OSError, AttributeError):
    _lib = None
    _function = None
    _HAS_GO_SC_SIGMA_DELTA_ACCUMULATOR = False


def simulate_sc_sigma_delta_accumulator(*args: object) -> dict[str, object]:
    """Run the complete configured Go project trace."""
    if _function is None:
        raise RuntimeError("Go SC SigmaDelta shared library is unavailable")
    config = tuple(float(cast(float, v)) for v in args[:2])
    currents = np.ascontiguousarray(args[2], dtype=np.float64)
    if currents.ndim != 1:
        raise ValueError("Go SC SigmaDelta current must be one-dimensional")
    trace = np.empty(currents.size, dtype=np.float64)
    events = np.empty(currents.size, dtype=np.int64)
    final = np.empty(1, dtype=np.float64)
    code = _function(
        ctypes.c_int(currents.size),
        *(ctypes.c_double(v) for v in config),
        ctypes.c_void_p(currents.ctypes.data),
        ctypes.c_void_p(trace.ctypes.data),
        ctypes.c_void_p(events.ctypes.data),
        ctypes.c_void_p(final.ctypes.data),
    )
    if code != 0:
        raise FloatingPointError(f"Go SC SigmaDelta batch failed with status {code}")
    return {"sigma": trace, "events": events, "sigma_final": float(final[0])}


__all__ = ["_HAS_GO_SC_SIGMA_DELTA_ACCUMULATOR", "simulate_sc_sigma_delta_accumulator"]
