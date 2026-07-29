# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ctypes facade for the Mojo Compte batch

"""Load and execute the complete Mojo Compte C ABI."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

_LIBRARY = Path(__file__).with_name("libcompte_wm.so")
try:
    _loaded_library = ctypes.CDLL(str(_LIBRARY))
    _loaded_function: Any = _loaded_library.compte_wm_simulate_c
    _loaded_function.restype = ctypes.c_int
    _lib: ctypes.CDLL | None = _loaded_library
    _function: Any | None = _loaded_function
    _HAS_MOJO_COMPTE_WM = True
except (OSError, AttributeError):
    _lib = None
    _function = None
    _HAS_MOJO_COMPTE_WM = False

_TRACE_KEYS = ("voltages", "s_ampa", "s_nmda", "x_nmda", "s_gaba", "refractory")
_FINAL_KEYS = (
    "v_final",
    "s_ampa_final",
    "s_nmda_final",
    "x_nmda_final",
    "s_gaba_final",
    "ref_final",
)


def simulate_compte_wm(*args: object) -> dict[str, object]:
    """Run one complete Mojo batch and expose only status-zero outputs."""
    if _function is None:
        raise RuntimeError("Mojo Compte shared library is unavailable")
    config = tuple(float(cast(float, value)) for value in args[:24])
    inputs = (
        np.ascontiguousarray(cast(npt.ArrayLike, args[24]), dtype=np.float64),
        *(
            np.ascontiguousarray(cast(npt.ArrayLike, value), dtype=np.int64)
            for value in args[25:28]
        ),
    )
    steps = inputs[0].size
    traces = {key: np.empty(steps, dtype=np.float64) for key in _TRACE_KEYS}
    events = np.empty(steps, dtype=np.int64)
    finals = {key: np.empty(1, dtype=np.float64) for key in _FINAL_KEYS}
    code = _function(
        ctypes.c_ssize_t(steps),
        *(ctypes.c_double(value) for value in config),
        *(ctypes.c_ssize_t(value.ctypes.data) for value in inputs),
        *(ctypes.c_ssize_t(traces[key].ctypes.data) for key in _TRACE_KEYS),
        ctypes.c_ssize_t(events.ctypes.data),
        *(ctypes.c_ssize_t(finals[key].ctypes.data) for key in _FINAL_KEYS),
    )
    if code != 0:
        raise FloatingPointError(f"Mojo Compte batch failed with status {code}")
    return {
        **traces,
        "events": events,
        **{key: float(value[0]) for key, value in finals.items()},
    }


__all__ = ["_HAS_MOJO_COMPTE_WM", "simulate_compte_wm"]
