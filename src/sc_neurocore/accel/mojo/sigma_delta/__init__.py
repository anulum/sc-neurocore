# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Load the executable Mojo sampled APSDM ABI."""

from __future__ import annotations
import ctypes
from pathlib import Path
from typing import Any, cast
import numpy as np

_LIBRARY = Path(__file__).with_name("libsigma_delta.so")
try:
    _loaded = ctypes.CDLL(str(_LIBRARY))
    _loaded_function: Any = _loaded.sigma_delta_simulate_c
    _loaded_function.restype = ctypes.c_int
    _lib: ctypes.CDLL | None = _loaded
    _function: Any | None = _loaded_function
    _HAS_MOJO_SIGMA_DELTA = True
except (OSError, AttributeError):
    _lib = None
    _function = None
    _HAS_MOJO_SIGMA_DELTA = False


def simulate_sigma_delta(*args: object) -> dict[str, object]:
    """Run a configured Mojo sampled APSDM trace."""
    if _function is None:
        raise RuntimeError("Mojo SigmaDelta shared library is unavailable")
    config = tuple(float(cast(float, v)) for v in args[:5])
    currents = np.ascontiguousarray(args[5], dtype=np.float64)
    traces = [np.empty(currents.size, dtype=np.float64) for _ in range(2)]
    events = np.empty(currents.size, dtype=np.int64)
    finals = [np.empty(1, dtype=np.float64) for _ in range(2)]
    code = _function(
        ctypes.c_ssize_t(currents.size),
        *(ctypes.c_double(v) for v in config),
        ctypes.c_ssize_t(currents.ctypes.data),
        *(ctypes.c_ssize_t(v.ctypes.data) for v in traces),
        ctypes.c_ssize_t(events.ctypes.data),
        *(ctypes.c_ssize_t(v.ctypes.data) for v in finals),
    )
    if code != 0:
        raise FloatingPointError(f"Mojo SigmaDelta batch failed with status {code}")
    return {
        "sigma": traces[0],
        "reconstruction": traces[1],
        "events": events,
        "sigma_final": float(finals[0][0]),
        "reconstruction_final": float(finals[1][0]),
    }


__all__ = ["_HAS_MOJO_SIGMA_DELTA", "simulate_sigma_delta"]
