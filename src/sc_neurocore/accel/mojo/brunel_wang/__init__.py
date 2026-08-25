# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ctypes facade for the Mojo Brunel-Wang batch

"""Load the executable Mojo C ABI without surrogate substitution."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, cast

import numpy as np

_LIBRARY = Path(__file__).with_name("libbrunel_wang.so")
try:
    _lib = ctypes.CDLL(str(_LIBRARY))
    function: Any = _lib.brunel_wang_simulate_c
    function.restype = ctypes.c_int
    _function: Any | None = function
    _HAS_MOJO_BRUNEL_WANG = True
except (OSError, AttributeError):
    _function = None
    _HAS_MOJO_BRUNEL_WANG = False


def simulate_brunel_wang(*args: object) -> dict[str, object]:
    """Run the complete Mojo batch and expose only status-zero outputs."""
    if _function is None:
        raise RuntimeError("Mojo Brunel-Wang shared library is unavailable")
    config = tuple(float(cast(float | int, value)) for value in args[:17])
    gates = tuple(np.ascontiguousarray(value, dtype=np.float64) for value in args[17:21])
    steps = gates[0].size
    voltages = np.empty(steps, dtype=np.float64)
    refractory = np.empty(steps, dtype=np.float64)
    events = np.empty(steps, dtype=np.int64)
    v_final = np.empty(1, dtype=np.float64)
    ref_final = np.empty(1, dtype=np.float64)
    code = _function(
        ctypes.c_ssize_t(steps),
        *(ctypes.c_double(value) for value in config),
        *(ctypes.c_ssize_t(value.ctypes.data) for value in gates),
        ctypes.c_ssize_t(voltages.ctypes.data),
        ctypes.c_ssize_t(refractory.ctypes.data),
        ctypes.c_ssize_t(events.ctypes.data),
        ctypes.c_ssize_t(v_final.ctypes.data),
        ctypes.c_ssize_t(ref_final.ctypes.data),
    )
    if code != 0:
        raise FloatingPointError(f"Mojo Brunel-Wang batch failed with status {code}")
    return {
        "voltages": voltages,
        "refractory": refractory,
        "events": events,
        "v_final": float(v_final[0]),
        "ref_final": float(ref_final[0]),
    }


__all__ = ["_HAS_MOJO_BRUNEL_WANG", "simulate_brunel_wang"]
